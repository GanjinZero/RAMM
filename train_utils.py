import torch
from torch import nn
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms
from PIL import Image
from randaugment import RandomAugment
from transformers import (
    AdamW, 
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from model.fusion import FusionVisualTextModel
from model.albef_fusion import AlbefFusionVisualTextModel
from model.meter_fusion import MeterFusionVisualTextModel, MeterFusionVisualTextPretraining
from model.meter_fusion import MeterFusionVisualTextRetrievalModel, MeterFusionVisualTextRetrievalModelV2
from model.mplug import MplugFusionVisualTextModel

from vqa_med_dataset import vqa_med_cls_dataset, vqa_rad_cls_dataset, slake_cls_dataset, vqa_med_2019_cls_dataset
from pretrain_dataset import pretrain_dataset
from vqa_med_dataset import retreive_dataset
from build_faiss_index import get_faiss_index


def generate_model(model_config, train_dataset, debug):
    if 'text_encoder' in model_config:
        return generate_visual_text_model(model_config, train_dataset, debug)
    return generate_visual_only_model(model_config, train_dataset, debug)

def generate_visual_text_model(model_config, train_dataset, debug):
    if model_config['fusion'] in ["concat", "mfh"]:
        model = FusionVisualTextModel(model_config, train_dataset, debug)
    if model_config['fusion'] == "albef":
        model = AlbefFusionVisualTextModel(model_config, train_dataset, debug)
    if model_config['fusion'] == "meter":
        if 'retrieval' not in model_config or not model_config['retrieval']:
            model = MeterFusionVisualTextModel(model_config, train_dataset, debug)
        else:
            if 'retrievalv2' in model_config and model_config['retrievalv2']:
                model = MeterFusionVisualTextRetrievalModelV2(model_config, train_dataset, debug)
            else:
                model = MeterFusionVisualTextRetrievalModel(model_config, train_dataset, debug)
    if model_config['fusion'] == "mplug":
        model = MplugFusionVisualTextModel(model_config, train_dataset, debug)
    return model

def generate_visual_only_model(model_config, train_dataset, debug):
    model = timm.create_model(model_config['backbone'], 
                              pretrained=not debug,
                              num_classes=train_dataset.num_answers,
                              img_size=model_config['image_res'])
    return model

def generate_pretrain_model(model_config, train_dataset, debug):
    return generate_pretrain_visual_text_model(model_config, train_dataset, debug)

def generate_pretrain_visual_text_model(model_config, train_dataset, debug):
    if model_config['fusion'] == "meter":
        model = MeterFusionVisualTextPretraining(model_config, train_dataset, debug)
    else:
        raise NotImplementedError
    return model

def create_transform(config):
    if config['transform'] == 'albef':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])  
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])   
        return train_transform, test_transform
    elif config['transform'] == 'sysu':
        train_transform = transforms.Compose([
            transforms.FixedResize(size=(config['image_res'] + 8, config['image_res'] + 8)),
            transforms.RandomCrop(size=(config['image_res'], config['image_res'])),
            # transforms.colorjitter_sample(parameters=(0.2, 0.2, 0.2, 0.)),
            transforms.RandomHorizontalFlip(),
            # transforms.Grayscale(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # transforms.Normalize(mean=(0.2565, 0.2564, 0.2564), std=(0.2223, 0.2224, 0.2222)),
            transforms.ToTensor()])

        test_transform = transforms.Compose([
            transforms.FixedResize(size=(config['image_res'], config['image_res'])),
            # transforms.Grayscale(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            # transforms.Normalize(mean=(0.2565, 0.2564, 0.2564), std=(0.2223, 0.2224, 0.2222)),
            transforms.ToTensor()])
        return train_transform, test_transform
    elif config['transform'] == 'noflip':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([                        
                transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),     
                transforms.ToTensor(),
                normalize,
            ])  
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
            ])   
        return train_transform, test_transform
    elif config['transform'] == 'clip_resizedcrop':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'], scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            transforms.CenterCrop(config['image_res']),
            transforms.ToTensor(),
            normalize,
        ])
        test_transform = transforms.Compose([
            transforms.Resize(config['image_res'], interpolation=Image.BICUBIC),
            transforms.CenterCrop(config['image_res']),
            transforms.ToTensor(),
            normalize,
        ])
        return train_transform, test_transform
    elif config['transform'] == 'clip':
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        test_transform = transforms.Compose([
            transforms.Resize(config['image_res'], interpolation=Image.BICUBIC),
            transforms.CenterCrop(config['image_res']),
            transforms.ToTensor(),
            normalize,
        ])
        return test_transform, test_transform

def not_n_in_list(n):
    for kw in ['visual_encoder', 'text_encoder', 'cross_modal', 'classifier']:
        if n.find(kw) >= 0:
            return False
    return True


def configure_optimizers(model, train_dataloader, config):
    if config['optimizer'].lower() == "adam":
        assert 'text_encoder' not in config
        optimizer = torch.optim.Adam(params, lr=config["learning_rate"])
    if config['optimizer'].lower() == "sgd":
        assert 'text_encoder' not in config
        optimizer = torch.optim.SGD(params, lr=config["learning_rate"], 
                                    momentum=config['momentum'],
                                    weight_decay=config['weight_decay'], nesterov=True)
    
    if config['optimizer'].lower() == "adamw":
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n.find('visual_encoder') >= 0],
                "weight_decay": config['weight_decay'],
                "lr": config["learning_rate"]
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find('visual_encoder') >= 0],
                "weight_decay": 0.0,
                "lr": config["learning_rate"]
            }, 
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n.find('text_encoder') >= 0],
                "weight_decay": config['weight_decay'],
                "lr": config.get("text_learning_rate", 1e-5)
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find('text_encoder') >= 0],
                "weight_decay": 0.0,
                "lr": config.get("text_learning_rate", 1e-5)
            }, 
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n.find('cross_modal') >= 0],
                "weight_decay": config['weight_decay'],
                "lr": config.get("cross_modal_learning_rate", 1e-4)
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find('cross_modal') >= 0],
                "weight_decay": 0.0,
                "lr": config.get("cross_modal_learning_rate", 1e-4)
            }, 
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and n.find('classifier') >= 0],
                "weight_decay": config['weight_decay'],
                "lr": config.get("classifier_learning_rate", 1e-4)
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and n.find('classifier') >= 0],
                "weight_decay": 0.0,
                "lr": config.get("classifier_learning_rate", 1e-4)
            }, 
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and not_n_in_list(n)],
                "weight_decay": config['weight_decay'],
                "lr": config["learning_rate"]
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and not_n_in_list(n)],
                "weight_decay": 0.0,
                "lr": config["learning_rate"]
            }, 
        ]

        
        optimizer = AdamW(params, eps=config['adam_epsilon'])
        
    total_steps = len(train_dataloader) * config['train_epoch']
    
    if 'scheduler' in config or config['scheduler'] == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config['warmup_ratio']),
            num_training_steps=total_steps,
        )
    elif config['scheduler'] == "constant":
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config['warmup_ratio'])
        )
    elif config['scheduler'] == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * config['warmup_ratio']),
            num_training_steps=total_steps,
        )
    
    return optimizer, scheduler

def create_dataset(data_config, train_transform, test_transform, debug=False):
    left_right_flip = data_config.get("left_right_flip", False)
    if left_right_flip:
        print("Left right Flip!")
    if data_config['dataset'] == "vqa_med":
        train_dataset = vqa_med_cls_dataset(data_config['train_file'], train_transform, split='train', answer_list=data_config['answer_list'], left_right_flip=left_right_flip)
        dev_dataset = None 
        test_dataset = vqa_med_cls_dataset(data_config['test_file'], test_transform, split='test', answer_list=data_config['answer_list'], left_right_flip=left_right_flip) 

    if data_config['dataset'] == "vqa_med_2019":
        train_dataset = vqa_med_2019_cls_dataset(data_config['train_file'], train_transform, split='train', answer_list=data_config['answer_list'], left_right_flip=left_right_flip)
        dev_dataset = vqa_med_2019_cls_dataset(data_config['dev_file'], train_transform, split='train', answer_list=data_config['answer_list'], left_right_flip=left_right_flip) 
        test_dataset = vqa_med_2019_cls_dataset(data_config['test_file'], test_transform, split='test', answer_list=data_config['answer_list'], left_right_flip=left_right_flip) 

    if data_config['dataset'] == "slake":
        train_dataset = slake_cls_dataset(data_config['train_file'], train_transform, split='train', answer_list=data_config['answer_list'], left_right_flip=left_right_flip)
        dev_dataset = slake_cls_dataset(data_config['dev_file'], train_transform, split='train', answer_list=data_config['answer_list'], left_right_flip=left_right_flip)
        test_dataset = slake_cls_dataset(data_config['test_file'], test_transform, split='test', answer_list=data_config['answer_list'], left_right_flip=left_right_flip) 

    if data_config['dataset'] == "vqa_rad":
        train_dataset = vqa_rad_cls_dataset(data_config['train_file'], train_transform, split='train', answer_list=data_config['answer_list'], left_right_flip=left_right_flip)
        dev_dataset = None 
        test_dataset = vqa_rad_cls_dataset(data_config['test_file'], test_transform, split='test', answer_list=data_config['answer_list'], left_right_flip=left_right_flip) 

    if data_config['retrieval']:
        if 'retrieval_range' in data_config:
            range = data_config['retrieval_range']
        else:
            range = 'fuse'
        if not debug:
            model, text_index, texts, fig_index, figs = get_faiss_index(data_config['faiss_model'], 
                                                                        range=range, debug=debug)
        else:
            model, text_index, texts, fig_index, figs = None, None, None, None, None
        train_dataset = retreive_dataset(train_dataset, retrieval_count=data_config['retrieval_count'], debug=debug,
                                         retrieval_by_rank=data_config['retrieval_by_rank'])
        train_dataset.retrieval(model, text_index, texts, fig_index, figs)
        if dev_dataset is not None:
            dev_dataset = retreive_dataset(dev_dataset, retrieval_count=data_config['retrieval_count'], debug=debug)
            dev_dataset.retrieval(model, text_index, texts, fig_index, figs)
        test_dataset = retreive_dataset(test_dataset, retrieval_count=data_config['retrieval_count'], debug=debug)
        test_dataset.retrieval(model, text_index, texts, fig_index, figs)

        if not debug:
            del model
            torch.cuda.empty_cache()

    return train_dataset, dev_dataset, test_dataset

def create_pretrain_dataset(data_config, train_transform):
    train_dataset = pretrain_dataset(data_config['train_file'], train_transform)
    return train_dataset

BERT_ABBV = {'Bio_ClinicalBERT':'clinicalbert',
             'bert-base-multilingual-cased': 'mbert_cased',
             'bert-base-multilingual-uncased': 'mbert_uncased',
             'bluebert_pubmed_mimic_base': 'bluebert',
             'bert-base-cased': 'base_cased',
             'bert-large-cased': 'large_cased',
             'bert-base-uncased': 'base_uncased',
             'bert-large-uncased': 'large_uncased',
             'pubmedbert_abs': 'pubmedbert',
             'scibert_scivocab_uncased': 'scibert',
             'biobert_v1.1': 'biobert',
             'biobert-large-cased-v1.1': 'biobertL',
             'spanbert-large-cased': 'span_large'}

def text_encoder_short_name(bert_name_or_path):
    if bert_name_or_path.lower().find('kebio') >= 0:
        return 'kebio'
    if bert_name_or_path.find('/') == -1:
        if bert_name_or_path in BERT_ABBV:
            return BERT_ABBV[bert_name_or_path]
        return bert_name_or_path
    if bert_name_or_path[-1] == "/":
        bert_name_or_path = bert_name_or_path[:-1]
    name = bert_name_or_path.split('/')[-1]
    if name in BERT_ABBV:
        return BERT_ABBV[name]
    return name

def ckpt_short_name(ckpt_name_or_path):
    if ckpt_name_or_path.find('checkpoint_vqa_7937.pth') >= 0:
        return ''

def generate_output_folder_name(data_config, model_config, debug):
    data_name = data_config['tag']
    epoch = model_config["train_epoch"]
    res = model_config["image_res"]
    model_name = model_config['backbone'].split('/')[0] + "_" + model_config["transform"] + "_" + f"{epoch}epoch" + "_" + f"{res}res"
    bsz = model_config["batch_size_train"] * model_config["gradient_accumulation_steps"]
    model_name = model_name + f"bsz{bsz}"
    if 'fusion' in model_config and 'text_encoder' in model_config:
        if model_config['fusion'] != "meter" and model_config['num_top_layer'] != 6:
            model_name = model_name + "_" + model_config["fusion"] + "_" + text_encoder_short_name(model_config["text_encoder"])
        else:
            model_name = model_name + "_" + "meter" + str(model_config['num_top_layer']) + "_" + text_encoder_short_name(model_config["text_encoder"])
    if 'learning_rate' in model_config:
        # if model_config['learning_rate'] != 1e-4:
        model_name = model_name + "_lr" + str(model_config['learning_rate'])
    
    if 'text_learning_rate' in model_config:
        # if model_config['text_learning_rate'] != 1e-5:
        model_name = model_name + "_tlr" + str(model_config['text_learning_rate'])
    if 'cross_modal_learning_rate' in model_config:
        model_name = model_name + "_xlr" + str(model_config['cross_modal_learning_rate'])
    if 'classifier_learning_rate' in model_config:
        model_name = model_name + "_clr" + str(model_config['classifier_learning_rate'])
    if 'left_right_flip' in model_config:
        if model_config['left_right_flip']:
            model_name = model_name + "_flip"
    if 'tag' in model_config and model_config['tag']:
        model_name = model_name + "_" + model_config['tag']
    if not 'loss' in model_config or model_config['loss'] == 'ce':
        pass
    else:
        if model_config['loss'] == "label_smooth":
            model_name = model_name + "_ls"
    if 'retrieval' in model_config and model_config['retrieval']:
        if 'retrieval_range' in data_config:
            ret_range = data_config['retrieval_range']
        else:
            ret_range = 'fuse'
        if 'retrieval_by_rank' in data_config and data_config['retrieval_by_rank']:
            retrieval_by_rank_str = '_rank'
        else:
            retrieval_by_rank_str = ''
        if 'retrievalv2' in model_config and model_config['retrievalv2']:
            name = 'retv2-'
        else:
            name = 'ret-'
        ret_cnt = model_config['retrieval_count']
        model_name = model_name + f"_{name}{ret_cnt}_{ret_range}{retrieval_by_rank_str}" 
    if 'rdrop' in model_config:
        rdrop = model_config['rdrop']
        model_name = model_name + f"_rdrop{rdrop}"
    if 'ema' in model_config:
        ema = model_config['ema']
        model_name = model_name + f"_ema{ema}"
    if 'ckpt' in model_config and model_config['ckpt']:
        if not 'ckpt_tag' in model_config:
            model_name = model_name + "_" + ckpt_short_name(model_config['ckpt'])
        else:
            model_name = model_name + "_" + model_config['ckpt_tag']
    # TODO: Add timestamp
    name = data_name + "_" + model_name
    if 'tag' in model_config and model_config["tag"]:
        name = name + "_" + model_config["tag"]
    if debug:
        name = name + "_" + "debug"
    return name

def generate_pretrain_output_folder_name(data_config, model_config, debug):
    return "pretrain_" + generate_output_folder_name(data_config, model_config, debug)
