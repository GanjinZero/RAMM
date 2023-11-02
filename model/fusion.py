import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import AutoModel, AutoConfig, AutoTokenizer
from model.mplug import mplug_create_model


class FusionVisualTextModel(nn.Module):
    def __init__(self, model_config, train_dataset, debug):
        super(FusionVisualTextModel, self).__init__()
        self.model_config = model_config

        if model_config['backbone'] == 'mplug':
            self.visual_encoder = mplug_create_model(model_config['ckpt'])
        else:
            self.visual_encoder = timm.create_model(model_config['backbone'], pretrained=not debug, num_classes=0, img_size=model_config['image_res'])
        self.text_encoder = AutoModel.from_pretrained(model_config['text_encoder'])
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_config['text_encoder'])

        self.visual_feat_dim = self.visual_encoder.num_features
        self.text_feat_dim = AutoConfig.from_pretrained(model_config['text_encoder']).hidden_size
        self.hidden_dim = max(self.visual_feat_dim, self.text_feat_dim)

        if model_config['fusion'] == "concat":
            self.classifier = nn.Sequential(nn.Linear(self.visual_feat_dim + self.text_feat_dim, self.hidden_dim), 
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_dim, train_dataset.num_answers))

        if model_config['fusion'] == "mfh":
            self.MFB_FACTOR_NUM = 5
            self.MFB_OUT_DIM = 1000
            self.JOINT_EMB_SIZE = self.MFB_FACTOR_NUM * self.MFB_OUT_DIM
            self.text_linear1 = nn.Linear(self.text_feat_dim, self.JOINT_EMB_SIZE)
            self.text_linear2 = nn.Linear(self.text_feat_dim, self.JOINT_EMB_SIZE)
            self.visual_linear1 = nn.Linear(self.visual_feat_dim, self.JOINT_EMB_SIZE)
            self.visual_linear2 = nn.Linear(self.visual_feat_dim, self.JOINT_EMB_SIZE)
            self.classifier = nn.Linear(self.MFB_OUT_DIM * 2, train_dataset.num_answers)
        
    def forward(self, image, text):
        if self.model_config['backbone'].find('vit') == -1:
            if self.model_config['backbone'].find('mplug') >= 0:
                image_hidden = self.visual_encoder(image)
            else:
                image_hidden = self.visual_encoder.forward_features(image)
        else:
            image_hidden = self.forward_visual_features(image)
            
        if len(image_hidden.shape) == 3:
            image_hidden = image_hidden[:,0]

        text = self.text_tokenizer(text, max_length=32, add_special_tokens=True,
                                   truncation=True, pad_to_max_length=True, return_tensors="pt")
        text_hidden = self.text_encoder(input_ids=text["input_ids"].to(image.device),
                                        attention_mask=text["attention_mask"].to(image.device))[1] # using CLS token

        if self.model_config['fusion'] == "concat":
            feat = torch.cat([image_hidden, text_hidden], dim=-1)
            logits = self.classifier(feat)

        if self.model_config['fusion'] == "mfh":
            mfb_q_o2_proj = self.text_linear1(text_hidden)                       # data_out (N, 5000)
            mfb_i_o2_proj = self.visual_linear1(image_hidden)              # img_feature (N, 5000)
            mfb_iq_o2_eltwise = torch.mul(mfb_q_o2_proj, mfb_i_o2_proj)
            mfb_iq_o2_resh = mfb_iq_o2_eltwise.view(-1, 1, self.MFB_OUT_DIM, self.MFB_FACTOR_NUM)  # N x 1 x 1000 x 5
            mfb_o2_out = torch.squeeze(torch.sum(mfb_iq_o2_resh, 3))                            # N x 1000
            mfb_o2_out = torch.sqrt(F.relu(mfb_o2_out)) - torch.sqrt(F.relu(-mfb_o2_out))       # signed sqrt
            mfb_o2_out = F.normalize(mfb_o2_out)

            mfb_q_o3_proj = self.text_linear2(text_hidden)                   # data_out (N, 5000)
            mfb_i_o3_proj = self.visual_linear2(image_hidden)          # img_feature (N, 5000)
            mfb_iq_o3_eltwise = torch.mul(mfb_q_o3_proj, mfb_i_o3_proj)
            mfb_iq_o3_eltwise = torch.mul(mfb_iq_o3_eltwise, mfb_iq_o2_eltwise)
            mfb_iq_o3_resh = mfb_iq_o3_eltwise.view(-1, 1, self.MFB_OUT_DIM, self.MFB_FACTOR_NUM)
            mfb_o3_out = torch.squeeze(torch.sum(mfb_iq_o3_resh, 3))                            # N x 1000
            mfb_o3_out = torch.sqrt(F.relu(mfb_o3_out)) - torch.sqrt(F.relu(-mfb_o3_out))
            mfb_o3_out = F.normalize(mfb_o3_out)

            mfb_o23_out = torch.cat((mfb_o2_out, mfb_o3_out), 1)        #200,2000     
            logits = self.classifier(mfb_o23_out) 

        return logits
