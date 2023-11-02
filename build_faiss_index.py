import os
from tqdm import trange
import json
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


overwrite_cache = False
batch_size = 16
device = "cuda:0"

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform = transforms.Compose([
    transforms.Resize((224, 224),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
    ])   


def build_index(embeddings, faiss_path):
    d = embeddings.shape[1]
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    faiss.write_index(index, faiss_path)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    return gpu_index

def get_text_embeddings(model, texts):
    model.eval()
    index = 0
    text_feats = []
    with tqdm(total=len(texts)) as pbar:
        while index < len(texts):
            now_texts = texts[index:index + batch_size]
            text = model.text_tokenizer(now_texts, max_length=100, add_special_tokens=True,
                                       truncation=True, pad_to_max_length=True, return_tensors="pt")
            text_input_ids = text["input_ids"].to(device)
            text_att_mask = text["attention_mask"].to(device)
            text_embeds = model.text_encoder(input_ids=text_input_ids,
                                            attention_mask=text_att_mask)[0]
            text_feat = F.normalize(model.text_proj(text_embeds[:,0,:]),dim=-1) 
            text_feats.append(text_feat.detach().cpu())
            pbar.update(min(index + batch_size, len(texts)) - index)
            index = min(index + batch_size, len(texts))
    return torch.cat(text_feats).numpy()


def get_fig_embeddings(model, figs):
    model.eval()
    index = 0
    fig_feats = []
    ava_index = None
    with tqdm(total=len(figs)) as pbar:
        while index < len(figs):
            now_figs = figs[index:index + batch_size]

            image = torch.stack([transform(Image.open(f).convert('RGB')).to(device) for f in now_figs], dim=0)
            #print('run')
            if model.model_config['backbone'].find('vit') == -1:
                if model.model_config['backbone'].find('mplug') >= 0:
                    image_hidden = model.visual_encoder(image)
                else:
                    image_hidden = model.visual_encoder.forward_features(image)
            else:
                image_hidden = model.forward_visual_features(image)
            image_embeds = model.visual_fc(image_hidden)
            fig_feat = F.normalize(model.vision_proj(image_embeds[:,0,:]),dim=-1)
            fig_feats.append(fig_feat.detach().cpu())
            pbar.update(min(index + batch_size, len(figs)) - index)
            index = min(index + batch_size, len(figs))
    return torch.cat(fig_feats).numpy(), ava_index


def build_base_name(pretrained_model_path, range):
    pretrain_name = pretrained_model_path.split('/')[-2] + "_" + \
                    pretrained_model_path.split('/')[-1].split('.')[0]
    return pretrain_name + "_" + range

def get_files(range):
    if range == "pmcp":
        files = ['data/pmc_patients_multimodal_train.json']
    elif range == "modality":
        files = ['data/pmc_multimodal_train.json']
    elif range == "all":
        files = ['data/pmc_multimodal_train.json',
                 'data/pmc_patients_multimodal_train.json']
    elif range == "fuse":
        files = ['data/all_pretrain.json']
    elif range == "rococxr":
        files = ['data/rococxr.json']
    elif range == "roco":
        files = ['data/roco.json']
    return files

def get_texts(range, debug, ava_text_index=None):
    files = get_files(range)
    texts = []
    for file in files:
        with open(file, 'r') as f:
            lines = json.load(f)
        for line in lines:
            # if not os.path.exists(line["image"]):
            #     continue
            texts.append(line["caption"])
            if debug:
                if len(texts) > 1000:
                    return texts
    if ava_text_index is not None:
        ava_text_index = set(ava_text_index)
        texts = [t for idx, t in enumerate(texts) if idx in ava_text_index]
    return texts

def get_text_index(model, range, text_faiss_path, debug, ava_text_index=None):
    texts = get_texts(range, debug, ava_text_index)
    if os.path.exists(text_faiss_path) and not overwrite_cache:
        print(f'Load text index {text_faiss_path}.')
        index = faiss.read_index(text_faiss_path)
        res = faiss.StandardGpuResources()
        text_index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        print(f'Generate text index {text_faiss_path}.')
        text_embeddings = get_text_embeddings(model, texts)
        text_index = build_index(text_embeddings, text_faiss_path)
    return text_index, texts

def get_figs(range, debug):
    files = get_files(range)
    texts = []
    for file in files:
        with open(file, 'r') as f:
            lines = json.load(f)
        for line in lines:
            # if os.path.exists(line["image"]):
            if "image_name" in line:
                texts.append(line["image_name"])
            elif "image" in line:
                texts.append(line["image"])
            if debug:
                if len(texts) > 1000:
                    return texts
    return texts

def get_fig_index(model, range, fig_faiss_path, debug):
    figs = get_figs(range, debug)
    if os.path.exists(fig_faiss_path) and not overwrite_cache:
        print(f'Load fig index {fig_faiss_path}.')
        index = faiss.read_index(fig_faiss_path)
        res = faiss.StandardGpuResources()
        fig_index = faiss.index_cpu_to_gpu(res, 0, index)
        ava_index = None
    else:
        print(f'Generate fig index {fig_faiss_path}.')
        fig_embeddings, ava_index = get_fig_embeddings(model, figs)
        fig_index = build_index(fig_embeddings, fig_faiss_path)
    return fig_index, figs, ava_index

def get_faiss_index(pretrained_model_path, range='fuse', debug=False):
    base_name = build_base_name(pretrained_model_path, range)
    try:
        os.mkdir(f"index/{base_name}")
    except:
        pass
    text_faiss_path = os.path.join("index", base_name, "text.index")
    fig_faiss_path = os.path.join("index", base_name, "fig.index")

    if debug:
        text_faiss_path = text_faiss_path + '.debug'
        fig_faiss_path = fig_faiss_path + '.debug'

    model = torch.load(pretrained_model_path).to(device)

    fig_index, figs, ava_index = get_fig_index(model, range, fig_faiss_path, debug=debug)
    text_index, texts = get_text_index(model, range, text_faiss_path, debug=debug, ava_text_index=ava_index)
    
    return model, text_index, np.array(texts), fig_index, np.array(figs)

def retrieval_text_from_imageq(q_embed, text_index, k=4):
    sim, idx = text_index.search(q_embed, k=k)
    return sim, idx

def retrieval_image_from_imageq(q_embed, fig_index, k=4):
    sim, idx = fig_index.search(q_embed, k=k)
    return sim, idx

def retrieval(model, imageq, text_index, fig_index, texts, figs, k=4):
    if isinstance(imageq, str):
        imageq = [imageq]
    q_embed, _ = get_fig_embeddings(model, imageq)
    if len(q_embed.shape) == 1:
        q_embed = q_embed.reshape(1, -1)
    sim_0, idx_0 = retrieval_text_from_imageq(q_embed, text_index, k+1)
    sim_1, idx_1 = retrieval_image_from_imageq(q_embed, fig_index, k+1)
    all_sim = np.concatenate([sim_0[:,1:], sim_1[:,1:]], axis=-1)
    all_idx = np.concatenate([idx_0[:,1:], idx_1[:,1:]], axis=-1)
    return all_idx, all_sim, \
           [texts[all_idx[i]].tolist() for i in range(len(imageq))], \
           [figs[all_idx[i]].tolist() for i in range(len(imageq))]

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1]
    model_path = os.path.join('pretrain_outputs', sys.argv[1], "epoch30.pth")
    ret_range = sys.argv[2]
    model, text_index, texts, fig_index, figs = get_faiss_index(model_path, range=sys.argv[2], debug=False)

