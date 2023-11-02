import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoConfig, AutoTokenizer
from transformers.models.bert.modeling_bert import BertIntermediate, BertOutput, BertAttention, BertSelfOutput
from transformers.modeling_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from model.mplug import mplug_create_model
from opt_einsum import contract
import math


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class MLMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return self.LayerNorm(pooled_output)

class BertCrossLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None #past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            None,
            encoder_hidden_states,
            encoder_attention_mask,
            None,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertCrossRetLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        self.crossattention = BertAttention(config)
        # self.retcrossattention = BertRetAttention(config)
        self.retcrossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        """
        hidden: b * n * len * h
        encoder: b * n * p * h
        """
        bsz, n, l, h = hidden_states.shape
        _, _, patch, _ = encoder_hidden_states.shape

        hidden_states_reshape = hidden_states.reshape(bsz * n, l , h)
        # if attention_mask is not None:
        #     attention_mask_reshape = attention_mask.reshape(bsz * n, l)
        # else:
        #     attention_mask_reshape = None
        encoder_hidden_states_reshape = encoder_hidden_states.reshape(bsz * n, patch , h)
        # if encoder_attention_mask is not None:
        #     encoder_attention_mask_reshape = encoder_attention_mask.reshape(bsz * n, patch)
        # else:
        #     encoder_attention_mask_reshape = None

        #import ipdb; ipdb.set_trace()
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = None #past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states_reshape,
            attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            None,
            encoder_hidden_states_reshape,
            encoder_attention_mask,
            None,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

        ret_cross_attention_outputs = self.retcrossattention(
            attention_output.reshape(bsz, n, l, h)[:,:,0],
            attention_mask=None,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )

        attention_output = attention_output.reshape(bsz, n, l, h)
        attention_output[:,:,0] += ret_cross_attention_outputs[0]

        #import ipdb; ipdb.set_trace()

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        #import ipdb; ipdb.set_trace()

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class MeterFusionVisualTextModel(nn.Module):
    def __init__(self, model_config, train_dataset, debug):
        super(MeterFusionVisualTextModel, self).__init__()
        self.model_config = model_config

        if model_config['backbone'] == 'mplug':
            self.visual_encoder = mplug_create_model(model_config['ckpt'])
        elif model_config['backbone'].find('clip') >= 0:
            self.visual_encoder = AutoModel.from_pretrained(model_config['backbone']).vision_model
        else:
            if not 'ckpt' in model_config:
                pretrained = not debug
            else:
                pretrained = not model_config['ckpt']
            self.visual_encoder = timm.create_model(model_config['backbone'], pretrained=pretrained, num_classes=0, img_size=model_config['image_res'])
        
        self.text_encoder = AutoModel.from_pretrained(model_config['text_encoder'])
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config['text_encoder'])

        if model_config['backbone'].find('clip') >= 0:
            self.visual_feat_dim = AutoConfig.from_pretrained(model_config['backbone']).vision_config.hidden_size
        else:
            self.visual_feat_dim = self.visual_encoder.num_features
        self.text_config = AutoConfig.from_pretrained(model_config['text_encoder'])
        self.text_feat_dim = self.text_config.hidden_size

        self.visual_fc = nn.Sequential(nn.Linear(self.visual_feat_dim, self.text_feat_dim),
                                       nn.LayerNorm(self.text_feat_dim, eps=1e-12),
                                       nn.Dropout(0.1))

        self.token_type_embeddings = nn.Embedding(2, self.text_feat_dim)

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(self.text_config) for _ in range(model_config['num_top_layer'])])
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(self.text_config) for _ in range(model_config['num_top_layer'])])

        self.cross_modal_text_pooler = Pooler(self.text_feat_dim)
        self.cross_modal_image_pooler = Pooler(self.text_feat_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(2 * self.text_feat_dim, train_dataset.num_answers)

    def forward_visual_features(self, x):
        x = self.visual_encoder.patch_embed(x)
        x = self.visual_encoder._pos_embed(x)
        x = self.visual_encoder.blocks(x)
        x = self.visual_encoder.norm(x)
        return x

    def forward(self, image, text):
        if self.model_config['backbone'].find('vit') == -1:
            if self.model_config['backbone'].find('mplug') >= 0:
                image_hidden = self.visual_encoder(image)
            else:
                image_hidden = self.visual_encoder.forward_features(image)
        else:
            if self.model_config['backbone'].find('clip') >= 0:
                image_hidden = self.visual_encoder(image).last_hidden_state
            else:
                image_hidden = self.forward_visual_features(image)

        #import ipdb; ipdb.set_trace()
            
        image_hidden = self.visual_fc(image_hidden)
        image_atts = torch.ones(image_hidden.size()[:-1],dtype=torch.long).to(image.device)

        text = self.text_tokenizer(text, max_length=32, add_special_tokens=True,
                                   truncation=True, pad_to_max_length=True, return_tensors="pt")

        text_hidden = self.text_encoder(input_ids=text["input_ids"].to(image.device),
                                        attention_mask=text["attention_mask"].to(image.device))[0]

        text_hidden += self.token_type_embeddings(torch.zeros_like(text["input_ids"]).to(image.device))
        image_hidden += self.token_type_embeddings(torch.ones_like(image_atts).to(image.device))
        x, y = text_hidden.to(image.device), image_hidden.to(image.device)
        extend_text_masks = self.text_encoder.get_extended_attention_mask(text["attention_mask"], text["attention_mask"].size(), x.device).to(x.device)
        #print(image_atts.shape)
        extend_image_masks = self.text_encoder.get_extended_attention_mask(image_atts, image_atts.size(), x.device).to(x.device)
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(text_feats)
        avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
        cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        logits = self.classifier(cls_feats)

        return logits

class MeterFusionVisualTextRetrievalModel(nn.Module):
    def __init__(self, model_config, train_dataset, debug):
        super(MeterFusionVisualTextRetrievalModel, self).__init__()
        self.model_config = model_config

        if model_config['backbone'] == 'mplug':
            self.visual_encoder = mplug_create_model(model_config['ckpt'])
        else:
            if not 'ckpt' in model_config:
                pretrained = not debug
            else:
                pretrained = not model_config['ckpt']
            self.visual_encoder = timm.create_model(model_config['backbone'], pretrained=pretrained, num_classes=0, img_size=model_config['image_res'])
        
        self.text_encoder = AutoModel.from_pretrained(model_config['text_encoder'])
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config['text_encoder'])

        self.visual_feat_dim = self.visual_encoder.num_features
        self.text_config = AutoConfig.from_pretrained(model_config['text_encoder'])
        self.text_feat_dim = self.text_config.hidden_size

        self.visual_fc = nn.Sequential(nn.Linear(self.visual_feat_dim, self.text_feat_dim),
                                       nn.LayerNorm(self.text_feat_dim, eps=1e-12),
                                       nn.Dropout(0.1))

        self.token_type_embeddings = nn.Embedding(2, self.text_feat_dim)
        # self.main_retrieve_embeddings = nn.Embedding(2, self.text_feat_dim)
        self.retrieve_id_embeddings = nn.Embedding(100, self.text_feat_dim)

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(self.text_config) for _ in range(model_config['num_top_layer'])])
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(self.text_config) for _ in range(model_config['num_top_layer'])])

        self.cross_modal_text_pooler = Pooler(self.text_feat_dim)
        self.cross_modal_image_pooler = Pooler(self.text_feat_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(2 * self.text_feat_dim, train_dataset.num_answers)

    def forward_visual_features(self, x):
        x = self.visual_encoder.patch_embed(x)
        x = self.visual_encoder._pos_embed(x)
        x = self.visual_encoder.blocks(x)
        x = self.visual_encoder.norm(x)
        return x

    def forward(self, image, text):
        # """
        # just for debug
        # """
        # duplicate_cnt = 3
        # image = image.unsqueeze(1).repeat(1,duplicate_cnt,1,1,1)
        # text = [[t] * duplicate_cnt for t in text]

        # image: b * n * ()
        # text: b * n * ()
        #import ipdb; ipdb.set_trace()
        image_shape = image.shape
        image_reshape = image.reshape((image.shape[0] * image.shape[1], *image.shape[2:]))
        if self.model_config['backbone'].find('vit') == -1:
            if self.model_config['backbone'].find('mplug') >= 0:
                image_hidden = self.visual_encoder(image_reshape)
            else:
                image_hidden = self.visual_encoder.forward_features(image_reshape)
        else:
            image_hidden = self.forward_visual_features(image_reshape)
            
        image_hidden = self.visual_fc(image_hidden) # (b * n) * p * h
        image_hidden = image_hidden.reshape((image_shape[0], image_shape[1], *image_hidden.shape[1:]))

        bsz, cnt, patch, hidden = image_hidden.shape
        image_atts = torch.ones(image_hidden.size()[:-1],dtype=torch.long).to(image.device)
        image_hidden += self.token_type_embeddings(torch.ones_like(image_atts).to(image.device))

        reshape_text = []
        for t in text:
            reshape_text.extend(t)
        text = self.text_tokenizer(reshape_text, max_length=32, add_special_tokens=True,
                                   truncation=True, pad_to_max_length=True, return_tensors="pt")
        text_hidden = self.text_encoder(input_ids=text["input_ids"].to(image.device),
                                        attention_mask=text["attention_mask"].to(image.device))[0] # (b * n) * len * h
        text_hidden += self.token_type_embeddings(torch.zeros_like(text["input_ids"]).to(image.device))
        length = text_hidden.shape[1]
        text_hidden = text_hidden.reshape(bsz, cnt, length, hidden) # b * n * len * h
       
        # add retrieve id embedding
        cnt_idx = torch.arange(cnt).to(image.device)
        image_hidden += self.retrieve_id_embeddings(cnt_idx).unsqueeze(1).unsqueeze(0)
        image_hidden = image_hidden.reshape(bsz, (cnt * patch), hidden)
        text_hidden += self.retrieve_id_embeddings(cnt_idx).unsqueeze(1).unsqueeze(0)
        text_hidden = text_hidden.reshape(bsz, (cnt * length), hidden)

        x, y = text_hidden.to(image.device), image_hidden.to(image.device)
        extend_text_masks = self.text_encoder.get_extended_attention_mask(text["attention_mask"].reshape(bsz, (cnt * length)), (bsz, (cnt * length)), x.device).to(x.device)
        extend_image_masks = self.text_encoder.get_extended_attention_mask(image_atts.reshape(bsz, (cnt * patch)), (bsz, (cnt * patch)), x.device).to(x.device)
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(text_feats)
        avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
        cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        logits = self.classifier(cls_feats)

        return logits
    

class MeterFusionVisualTextRetrievalModelV2(nn.Module):
    def __init__(self, model_config, train_dataset, debug):
        super(MeterFusionVisualTextRetrievalModelV2, self).__init__()
        self.model_config = model_config

        if model_config['backbone'] == 'mplug':
            self.visual_encoder = mplug_create_model(model_config['ckpt'])
        else:
            if not 'ckpt' in model_config:
                pretrained = not debug
            else:
                pretrained = not model_config['ckpt']
            self.visual_encoder = timm.create_model(model_config['backbone'], pretrained=pretrained, num_classes=0, img_size=model_config['image_res'])
        
        self.text_encoder = AutoModel.from_pretrained(model_config['text_encoder'])
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config['text_encoder'])

        self.visual_feat_dim = self.visual_encoder.num_features
        self.text_config = AutoConfig.from_pretrained(model_config['text_encoder'])
        self.text_feat_dim = self.text_config.hidden_size

        self.visual_fc = nn.Sequential(nn.Linear(self.visual_feat_dim, self.text_feat_dim),
                                       nn.LayerNorm(self.text_feat_dim, eps=1e-12),
                                       nn.Dropout(0.1))

        self.retrieve_id_embeddings = nn.Embedding(100, self.text_feat_dim)

        self.cross_modal_image_layers = nn.ModuleList([BertCrossRetLayer(self.text_config) for _ in range(model_config['num_top_layer'])])
        self.cross_modal_text_layers = nn.ModuleList([BertCrossRetLayer(self.text_config) for _ in range(model_config['num_top_layer'])])

        self.cross_modal_text_pooler = Pooler(self.text_feat_dim)
        self.cross_modal_image_pooler = Pooler(self.text_feat_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(2 * self.text_feat_dim, train_dataset.num_answers)

    def forward_visual_features(self, x):
        x = self.visual_encoder.patch_embed(x)
        x = self.visual_encoder._pos_embed(x)
        x = self.visual_encoder.blocks(x)
        x = self.visual_encoder.norm(x)
        return x

    def forward(self, image, text):
        image_shape = image.shape
        image_reshape = image.reshape((image.shape[0] * image.shape[1], *image.shape[2:]))
        if self.model_config['backbone'].find('vit') == -1:
            if self.model_config['backbone'].find('mplug') >= 0:
                image_hidden = self.visual_encoder(image_reshape)
            else:
                image_hidden = self.visual_encoder.forward_features(image_reshape)
        else:
            image_hidden = self.forward_visual_features(image_reshape)
            
        image_hidden = self.visual_fc(image_hidden) # (b * n) * p * h
        image_hidden = image_hidden.reshape((image_shape[0], image_shape[1], *image_hidden.shape[1:]))

        bsz, cnt, patch, hidden = image_hidden.shape

        image_atts = torch.ones(image_hidden.size()[:-1],dtype=torch.long).to(image.device)

        reshape_text = []
        for t in text:
            reshape_text.extend(t)
        text = self.text_tokenizer(reshape_text, max_length=32, add_special_tokens=True,
                                   truncation=True, pad_to_max_length=True, return_tensors="pt")
        text_hidden = self.text_encoder(input_ids=text["input_ids"].to(image.device),
                                        attention_mask=text["attention_mask"].to(image.device))[0] # (b * n) * len * h
        length = text_hidden.shape[1]
        text_hidden = text_hidden.reshape(bsz, cnt, length, hidden) # b * n * len * h
       
        # add retrieve id embedding
        # cnt_idx = torch.arange(cnt).to(image.device)
        # image_hidden += self.retrieve_id_embeddings(cnt_idx).unsqueeze(1).unsqueeze(0)
        # image_hidden = image_hidden.reshape(bsz, (cnt * patch), hidden)
        # text_hidden += self.retrieve_id_embeddings(cnt_idx).unsqueeze(1).unsqueeze(0)
        # text_hidden = text_hidden.reshape(bsz, (cnt * length), hidden)

        x, y = text_hidden.to(image.device), image_hidden.to(image.device)
        extend_text_masks = self.text_encoder.get_extended_attention_mask(text["attention_mask"].reshape(bsz * cnt, length), (bsz * cnt, length), x.device).to(x.device)
        extend_image_masks = self.text_encoder.get_extended_attention_mask(image_atts.reshape(bsz * cnt, patch), (bsz * cnt, patch), x.device).to(x.device)
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(text_feats[:,0])
        image_feats = image_feats[:,0]
        avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
        cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        cls_feats = torch.cat([cls_feats_text, cls_feats_image], dim=-1)
        logits = self.classifier(cls_feats)

        return logits
    

class MeterFusionVisualTextPretraining(nn.Module):
    def __init__(self, model_config, train_dataset, debug):
        super(MeterFusionVisualTextPretraining, self).__init__()
        self.model_config = model_config

        if model_config['backbone'] == 'mplug':
            self.visual_encoder = mplug_create_model(model_config['ckpt'])
        else:
            self.visual_encoder = timm.create_model(model_config['backbone'], pretrained=not debug, num_classes=0, img_size=model_config['image_res'])
        
        self.text_encoder = AutoModel.from_pretrained(model_config['text_encoder'])
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config['text_encoder'])

        self.visual_feat_dim = self.visual_encoder.num_features
        self.text_config = AutoConfig.from_pretrained(model_config['text_encoder'])
        self.text_feat_dim = self.text_config.hidden_size

        self.visual_fc = nn.Sequential(nn.Linear(self.visual_feat_dim, self.text_feat_dim),
                                       nn.LayerNorm(self.text_feat_dim, eps=1e-12),
                                       nn.Dropout(0.1))

        self.token_type_embeddings = nn.Embedding(2, self.text_feat_dim)

        self.cross_modal_image_layers = nn.ModuleList([BertCrossLayer(self.text_config) for _ in range(model_config['num_top_layer'])])
        self.cross_modal_text_layers = nn.ModuleList([BertCrossLayer(self.text_config) for _ in range(model_config['num_top_layer'])])

        self.cross_modal_text_pooler = Pooler(self.text_feat_dim)
        self.cross_modal_image_pooler = Pooler(self.text_feat_dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.classifier = nn.Linear(2 * self.text_feat_dim, train_dataset.num_answers)

        # pretrain related
        self.temp = nn.Parameter(torch.ones([]) * model_config['temp'])   
        self.embed_dim = model_config['embed_dim']
        self.vision_proj = nn.Linear(self.text_feat_dim, self.embed_dim)
        self.text_proj = nn.Linear(self.text_feat_dim, self.embed_dim)

        self.queue_size = model_config['queue_size']
        self.momentum = model_config['momentum']  
        self.itm_head = nn.Linear(self.text_feat_dim * 2, 2)  
        self.mlm_pooler = MLMHead(self.text_feat_dim)

        self.mlm_probability = model_config['mlm_probability']

        # m nn
        if model_config['backbone'] == 'mplug':
            self.visual_encoder_m = mplug_create_model(model_config['ckpt'])
        else:
            self.visual_encoder_m = timm.create_model(model_config['backbone'], pretrained=not debug, num_classes=0, img_size=model_config['image_res'])
        self.text_encoder_m = AutoModel.from_pretrained(model_config['text_encoder'])
        self.visual_fc_m = nn.Sequential(nn.Linear(self.visual_feat_dim, self.text_feat_dim),
                                         nn.LayerNorm(self.text_feat_dim, eps=1e-12),
                                         nn.Dropout(0.1))
        self.token_type_embeddings_m = nn.Embedding(2, self.text_feat_dim)

        self.cross_modal_image_layers_m = nn.ModuleList([BertCrossLayer(self.text_config) for _ in range(model_config['num_top_layer'])])
        self.cross_modal_text_layers_m = nn.ModuleList([BertCrossLayer(self.text_config) for _ in range(model_config['num_top_layer'])])
        self.vision_proj_m = nn.Linear(self.text_feat_dim, self.embed_dim)
        self.text_proj_m = nn.Linear(self.text_feat_dim, self.embed_dim)
        self.mlm_pooler_m = MLMHead(self.text_feat_dim)
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.visual_fc, self.visual_fc_m],
                            [self.token_type_embeddings, self.token_type_embeddings_m],
                            [self.cross_modal_image_layers, self.cross_modal_image_layers_m],
                            [self.cross_modal_text_layers, self.cross_modal_text_layers_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_proj, self.text_proj_m],
                            [self.mlm_pooler, self.mlm_pooler_m]]

        # create the queue
        self.queue_size = model_config['queue_size']
        self.register_buffer("image_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.copy_params() 

    def forward_visual_features(self, x):
        x = self.visual_encoder.patch_embed(x)
        x = self.visual_encoder._pos_embed(x)
        x = self.visual_encoder.blocks(x)
        x = self.visual_encoder.norm(x)
        return x

    def forward_visual_features_m(self, x):
        x = self.visual_encoder_m.patch_embed(x)
        x = self.visual_encoder_m._pos_embed(x)
        x = self.visual_encoder_m.blocks(x)
        x = self.visual_encoder_m.norm(x)
        return x

    def forward(self, image, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        if self.model_config['backbone'].find('vit') == -1:
            if self.model_config['backbone'].find('mplug') >= 0:
                image_hidden = self.visual_encoder(image)
            else:
                image_hidden = self.visual_encoder.forward_features(image)
        else:
            image_hidden = self.forward_visual_features(image)
            
        image_embeds = self.visual_fc(image_hidden)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)

        text = self.text_tokenizer(text, max_length=100, add_special_tokens=True,
                                   truncation=True, pad_to_max_length=True, return_tensors="pt")
        text_input_ids = text["input_ids"].to(image.device)
        text_att_mask = text["attention_mask"].to(image.device)
        text_embeds = self.text_encoder(input_ids=text_input_ids,
                                        attention_mask=text_att_mask)[0]
        
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 
             
        # get momentum features
        with torch.no_grad():
            self._momentum_update()
            if self.model_config['backbone'].find('vit') == -1:
                if self.model_config['backbone'].find('mplug') >= 0:
                    image_embeds_m = self.visual_encoder_m(image)
                else:
                    image_embeds_m = self.visual_encoder_m.forward_features(image)
            else:
                image_embeds_m = self.forward_visual_features_m(image)
            image_embeds_m = self.visual_fc_m(image_embeds_m)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  
            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                                         
            text_output_m = self.text_encoder_m(input_ids=text_input_ids,
                                                attention_mask=text_att_mask)[0]  
            text_feat_m = F.normalize(self.text_proj_m(text_output_m[:,0,:]),dim=-1) 
            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)          

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_all / self.temp 
        sim_t2i = text_feat @ image_feat_all / self.temp 
                             
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        loss_ita = (loss_i2t + loss_t2i) / 2

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)

        ###=================================###
        # forward the positve image-text pair
        text_hidden_before_fuse = text_embeds + self.token_type_embeddings(torch.zeros_like(text_input_ids))
        image_hidden_before_fuse = image_embeds + self.token_type_embeddings(torch.ones_like(image_atts))
        x, y = text_hidden_before_fuse, image_hidden_before_fuse

        extend_text_masks = self.text_encoder.get_extended_attention_mask(text_att_mask, text_att_mask.size(), x.device)
        extend_image_masks = self.text_encoder.get_extended_attention_mask(image_atts, image_atts.size(), x.device)
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]

        text_feats, image_feats = x, y
        cls_feats_text = self.cross_modal_text_pooler(text_feats)
        avg_image_feats = self.avgpool(image_feats.transpose(1, 2)).view(image_feats.size(0), 1, -1)
        cls_feats_image = self.cross_modal_image_pooler(avg_image_feats)
        output_pos = torch.cat([cls_feats_text, cls_feats_image], dim=-1)

        with torch.no_grad():
            bs = image.size(0)          
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)
   
            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text
        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_att_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text_att_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg, image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts, image_atts],dim=0)

        text_hidden_before_fuse_neg = text_embeds_all + self.token_type_embeddings(torch.zeros(text_embeds_all.shape[0:-1]).long().to(image.device))
        image_hidden_before_fuse_neg = image_embeds_all + self.token_type_embeddings(torch.ones_like(image_atts_all))

        extend_text_masks_neg = self.text_encoder.get_extended_attention_mask(text_atts_all, text_atts_all.size(), x.device)
        extend_image_masks_neg = self.text_encoder.get_extended_attention_mask(image_atts_all, image_atts_all.size(), x.device)

        x, y = text_hidden_before_fuse_neg, image_hidden_before_fuse_neg
        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks_neg, extend_image_masks_neg)
            y1 = image_layer(y, x, extend_image_masks_neg, extend_text_masks_neg)
            x, y = x1[0], y1[0]

        text_feats_neg, image_feats_neg = x, y
        cls_feats_text_neg = self.cross_modal_text_pooler(text_feats_neg)
        avg_image_feats_neg = self.avgpool(image_feats_neg.transpose(1, 2)).view(image_feats_neg.size(0), 1, -1)
        cls_feats_image_neg = self.cross_modal_image_pooler(avg_image_feats_neg)
        output_neg = torch.cat([cls_feats_text_neg, cls_feats_image_neg], dim=-1)                   

        vl_embeddings = torch.cat([output_pos, output_neg],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)     
        
        ##================= MLM ========================##                
        input_ids = text_input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)                    
        input_ids, labels = self.mask(input_ids, self.text_config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix)
        #print(input_ids[0], labels[0])
        
        with torch.no_grad():
            text_hidden_before_mlm_m = self.text_encoder_m(input_ids=input_ids,
                                                         attention_mask=text_att_mask)[0] 
            text_hidden_before_mlm_m = text_hidden_before_mlm_m + self.token_type_embeddings_m(torch.zeros_like(input_ids))
            image_hidden_before_mlm_m = image_embeds_m + self.token_type_embeddings_m(torch.ones_like(image_atts))
            x, y = text_hidden_before_mlm_m, image_hidden_before_mlm_m

            for text_layer, image_layer in zip(self.cross_modal_text_layers_m, self.cross_modal_image_layers_m):
                x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
                y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
                x, y = x1[0], y1[0]
            mlm_rep_m = self.mlm_pooler_m(x) # batch * token * hidden
            mlm_logits_m = contract('blh,wh->blw', mlm_rep_m, self.text_encoder_m.embeddings.word_embeddings.weight)

        text_hidden_before_mlm = self.text_encoder(input_ids=input_ids,
                                                   attention_mask=text_att_mask)[0] 
        text_hidden_before_mlm = text_hidden_before_mlm + self.token_type_embeddings(torch.zeros_like(input_ids))
        image_hidden_before_mlm = image_hidden_before_fuse
        x, y = text_hidden_before_mlm, image_hidden_before_mlm

        for text_layer, image_layer in zip(self.cross_modal_text_layers, self.cross_modal_image_layers):
            x1 = text_layer(x, y, extend_text_masks, extend_image_masks)
            y1 = image_layer(y, x, extend_image_masks, extend_text_masks)
            x, y = x1[0], y1[0]
        mlm_rep = self.mlm_pooler(x) # batch * token * hidden
        mlm_logits = contract('blh,wh->blw', mlm_rep, self.text_encoder.embeddings.word_embeddings.weight)

        loss_fct = nn.CrossEntropyLoss()

        masked_lm_loss = loss_fct(mlm_logits.view(-1, mlm_logits.shape[-1]), labels.view(-1))
        loss_mlm_distill = - torch.sum(F.log_softmax(mlm_logits, dim=-1) * F.softmax(mlm_logits_m, dim=-1), dim=-1)
        loss_mlm_distill = loss_mlm_distill[labels!=-100].mean()
        loss_mlm = (1 - alpha) * masked_lm_loss + alpha * loss_mlm_distill
        return loss_mlm, loss_ita, loss_itm  
        #return loss_mlm, 0 * loss_ita, 0 * loss_itm

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr 
        
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:                                       
            masked_indices = torch.bernoulli(probability_matrix).bool()
                                               
        masked_indices[input_ids == self.text_tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.text_tokenizer.cls_token_id] = False
        
        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens            

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.text_tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]                     
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged   
        
        if targets is not None:
            return input_ids, targets
        else:
            return input_ids
        
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
