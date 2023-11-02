import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoConfig, AutoTokenizer
from model.mplug import mplug_create_model


class AlbefFusionVisualTextModel(nn.Module):
    def __init__(self, model_config, train_dataset, debug):
        super(AlbefFusionVisualTextModel, self).__init__()
        self.model_config = model_config

        if model_config['backbone'] == 'mplug':
            self.visual_encoder = mplug_create_model(model_config['ckpt'])
        else:
            self.visual_encoder = timm.create_model(model_config['backbone'], pretrained=not debug, num_classes=0, img_size=model_config['image_res'])        
        self.text_encoder = AutoModel.from_pretrained(model_config['text_encoder'])
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config['text_encoder'])

        self.visual_feat_dim = self.visual_encoder.num_features
        self.text_feat_dim = AutoConfig.from_pretrained(model_config['text_encoder']).hidden_size

        self.visual_fc = nn.Sequential(nn.Linear(self.visual_feat_dim, self.text_feat_dim),
                                       nn.LayerNorm(self.text_feat_dim, eps=1e-12),
                                       nn.Dropout(0.1))

        self.classifier = nn.Linear(self.text_feat_dim, train_dataset.num_answers)

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
            image_hidden = self.forward_visual_features(image)
        image_hidden = self.visual_fc(image_hidden)
        image_atts = torch.ones(image_hidden.size()[:-1],dtype=torch.long).to(image.device)

        text = self.text_tokenizer(text, max_length=32, add_special_tokens=True,
                                   truncation=True, pad_to_max_length=True, return_tensors="pt")

        text_hidden = self.text_encoder(input_ids=text["input_ids"].to(image.device),
                                        attention_mask=text["attention_mask"].to(image.device),
                                        encoder_hidden_states=image_hidden,
                                        encoder_attention_mask=image_atts)[1] # using CLS token

        logits = self.classifier(text_hidden)

        return logits
    
