from collections import OrderedDict
import torch
from torch import nn
from model import xbert
from transformers import AutoTokenizer, AutoConfig


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor, text_mask=None):
        if text_mask is None:
            text_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=text_mask)[0]

    def forward(self, x: torch.Tensor, text_mask = None):
        x = x + self.attention(self.ln_1(x), text_mask = text_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, text_mask=None):
        for layer in self.resblocks:
            #x = layer(x, text_mask = text_mask)
            x = torch.utils.checkpoint.checkpoint(layer, x , text_mask)
        return x


class MplugVisualTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, fix_embedding=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.heads = heads
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.fix_embedding = fix_embedding

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.num_features = width

    def forward(self, x: torch.Tensor, skip_last_layer=True):
        # print ("patch linear project")
        # print ("fix_embedding: ", self.fix_embedding)
        if self.fix_embedding:
            with torch.no_grad():
                x = self.conv1(x)  # shape = [*, width, grid, grid]
        else:
            x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)[:x.size(1),:]
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        
        if skip_last_layer:
            x = self.ln_post(x)
            # x = x @ self.proj
        else:         
            x = x @ self.proj
        return x

class MplugFusionVisualTextModel(nn.Module):
    def __init__(self, model_config, train_dataset, debug):
        super(MplugFusionVisualTextModel, self).__init__()
        self.model_config = model_config

        self.visual_encoder = mplug_create_model(model_config['ckpt'])
        self.text_encoder = xbert.BertModel.from_pretrained(model_config['text_tokenizer'], config=model_config['text_encoder'])
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config['text_tokenizer'])

        self.visual_feat_dim = self.visual_encoder.num_features
        self.text_config = AutoConfig.from_pretrained(model_config['text_encoder'])
        self.text_feat_dim = self.text_config.hidden_size

        self.visual_fc = nn.Sequential(nn.Linear(self.visual_feat_dim, self.text_feat_dim),
                                       nn.LayerNorm(self.text_feat_dim, eps=1e-12),
                                       nn.Dropout(0.1))

        self.classifier = nn.Linear(self.text_feat_dim, train_dataset.num_answers)

    def forward(self, image, text):
        image_hidden = self.visual_encoder(image)
        image_hidden = self.visual_fc(image_hidden)
        image_atts = torch.ones(image_hidden.size()[:-1],dtype=torch.long).to(image.device)

        text = self.text_tokenizer(text, max_length=32, add_special_tokens=True,
                                   truncation=True, pad_to_max_length=True, return_tensors="pt")

        question_output = self.text_encoder(text["input_ids"].to(image.device), 
                                        attention_mask = text["attention_mask"].to(image.device), 
                                        encoder_hidden_states = image_hidden,
                                        encoder_attention_mask = image_atts,                                    
                                        return_dict = True) 
        hidden = question_output.last_hidden_state[:,0]
        logits = self.classifier(hidden)
        return logits

class MplugMeterFusionVisualTextModel(nn.Module):
    def __init__(self, model_config, train_dataset, debug):
        super(MplugMeterFusionVisualTextModel, self).__init__()
        self.model_config = model_config

        self.visual_encoder = mplug_create_model(model_config['ckpt'])
        self.text_encoder = xbert.BertModel.from_pretrained(model_config['text_tokenizer'], config=model_config['text_encoder'])
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config['text_tokenizer'])

        self.visual_feat_dim = self.visual_encoder.num_features
        self.text_config = AutoConfig.from_pretrained(model_config['text_encoder'])
        self.text_feat_dim = self.text_config.hidden_size

        self.visual_fc = nn.Sequential(nn.Linear(self.visual_feat_dim, self.text_feat_dim),
                                       nn.LayerNorm(self.text_feat_dim, eps=1e-12),
                                       nn.Dropout(0.1))

        self.classifier = nn.Linear(self.text_feat_dim, train_dataset.num_answers)

    def forward(self, image, text):
        image_hidden = self.visual_encoder(image)
        image_hidden = self.visual_fc(image_hidden)
        image_atts = torch.ones(image_hidden.size()[:-1],dtype=torch.long).to(image.device)

        text = self.text_tokenizer(text, max_length=32, add_special_tokens=True,
                                   truncation=True, pad_to_max_length=True, return_tensors="pt")

        question_output = self.text_encoder(text["input_ids"].to(image.device), 
                                        attention_mask = text["attention_mask"].to(image.device), 
                                        encoder_hidden_states = image_hidden,
                                        encoder_attention_mask = image_atts,                                    
                                        return_dict = True) 
        hidden = question_output.last_hidden_state[:,0]
        logits = self.classifier(hidden)
        return logits


def mplug_create_model(state_dict_path):
    state_dict = torch.load(state_dict_path, map_location='cpu')['model']

    vision_width = state_dict["visual_encoder.visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual_encoder.visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual_encoder.visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual_encoder.visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["visual_encoder.text_projection"].shape[1]
    # context_length = state_dict["positional_embedding"].shape[0]
    # vocab_size = state_dict["token_embedding.weight"].shape[0]
    # transformer_width = state_dict["ln_final.weight"].shape[0]
    # transformer_heads = transformer_width // 64
    # transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    print(state_dict_path)
    print(f"image_resolution: {image_resolution}")
    print(f"vision_patch_size: {vision_patch_size}")
    print(f"vision_width: {vision_width}")
    print(f"vision_layers: {vision_layers}")
    print(f"embed_dim: {embed_dim}")

    vision_heads = vision_width // 64
    print(f"vision_heads: {vision_heads}")
    visual = MplugVisualTransformer(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        output_dim=embed_dim,
        fix_embedding=False
    )
    return visual