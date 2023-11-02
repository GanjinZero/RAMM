import os
import json
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import re
from torchvision import transforms
import torchvision.transforms.functional as F
#from dataset.vqa_med_dataset import AnswerTable
from tqdm import tqdm, trange
import numpy as np
from build_faiss_index import retrieval


def pre_question(question,max_ques_words):
    question = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        question.lower(),
    ).replace('-', ' ').replace('/', ' ')  
    question = question.rstrip(' ')
    
    #truncate question
    question_words = question.split(' ')
    if len(question_words)>max_ques_words:
        question = ' '.join(question_words[:max_ques_words])
            
    return question

def pre_answer(answer):
    answer = str(answer)
    answer = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        answer.lower(),
    ).replace('-', '').replace('/', '')  
    answer = answer.strip()
    return answer


class vqa_med_dataset(Dataset):
    def __init__(self, question_files, transform, eos='[SEP]', split="train", max_ques_words=30, answer_list=''):
        self.split = split        
        self.ann = []
        idx = 0
        for question_file in question_files:
            with open(question_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split('|')
                    pic_path = os.path.join("/".join(question_file.split('/')[0:-1]), 'images', line[0] + ".jpg")
                    self.ann += [{'image_name':pic_path, 'question':line[1], 'answer':line[2:], 'qid':idx}]
                    idx += 1

        self.transform = transform
        self.max_ques_words = max_ques_words
        self.eos = eos
        
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
            if answer_list:
                self.answer_list = json.load(open(answer_list,'r'))
            else:
                self.answer_list = ['yes', 'no']
                
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(ann['image_name'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        question = pre_question(ann['question'], self.max_ques_words) # can also use question_rephrase
        
        if self.split == 'test':
            question_id = ann['qid']            
            return image, question, question_id
        elif self.split=='train':                            
            answers = [ans + self.eos for ans in ann['answer']]
            weights = [1/len(answers)] * len(answers)
            return image, question, answers, weights

class vqg_med_dataset(Dataset):
    def __init__(self, question_files, query_files, transform, eos='[SEP]', split="train", max_ques_words=30, answer_list=None):
        self.split = split        
        self.ann = []
        idx = 0
        for question_file in question_files:
            with open(question_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split('|')
                    pic_path = os.path.join("/".join(question_file.split('/')[0:-1]), 'images', line[0] + ".jpg")
                    self.ann += [{'image_name':pic_path, 'question':'', 'answer':line[1:-1], 'qid':idx}]
                    idx += 1
        for question_file in query_files:
            with open(question_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split('|')
                    pic_path = os.path.join("/".join(question_file.split('/')[0:-1]), 'images', line[0] + ".jpg")
                    self.ann += [{'image_name':pic_path, 'question':'', 'answer':line[1:], 'qid':idx}]
                    idx += 1


        self.transform = transform
        self.max_ques_words = max_ques_words
        self.eos = eos
        
        if split=='test':
            self.max_ques_words = 50 # do not limit question length during test
            self.answer_list = ['yes', 'no'] # hard code
                
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(ann['image_name'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        question = pre_question(ann['question'], self.max_ques_words) # can also use question_rephrase
        
        if self.split == 'test':
            question_id = ann['qid']            
            return image, question, question_id
        elif self.split=='train':                            
            answers = [ans + self.eos for ans in ann['answer']]
            weights = [1/len(answers)] * len(answers)
            return image, question, answers, weights

def swap_words(s, x, y):
    return y.join(part.replace(y, x) for part in s.split(x))

def flip(x):
    if x.find('left') >= 0 and x.find('right') == -1:
        return x.replace('left', 'right')
    if x.find('right') >= 0 and x.find('left') == -1:
        return x.replace('right', 'left')
    if x.find('left') >= 0 and x.find('right') >= 0:
        return swap_words(x, 'left', 'right')
    return x

class vqa_med_cls_dataset(Dataset):
    def __init__(self, question_files, transform, eos='[SEP]', split="train", max_ques_words=30, answer_list='', left_right_flip=False):
        self.split = split        
        self.ann = []
        self.id2datum = {}

        idx = 0
        for question_file in question_files:
            with open(question_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split('|')
                    pic_path = os.path.join("/".join(question_file.split('/')[0:-1]), 'images', line[0] + ".jpg")
                    self.ann += [{'image_name':pic_path, 'question':line[1], 'answer':line[2:], 'qid':idx}]
                    idx += 1
                    self.id2datum[idx] = idx

        self.transform = transform
        self.max_ques_words = max_ques_words
        self.eos = eos

        self.left_right_flip = left_right_flip

        self.answer_list = json.load(open(answer_list, 'r'))
        self.ans2normal = {ans:pre_answer(ans) for ans in self.answer_list}
        #flip_list = []
        if self.left_right_flip:
            for ans in self.answer_list:
                pre_ans = pre_answer(ans)
                if flip(pre_ans) != pre_ans and not flip(pre_ans) in self.ans2normal.values():
                    self.ans2normal[flip(pre_ans)] = flip(pre_ans)
                    #flip_list.append(pre_ans)
        #print(flip_list)
        self.normal_list = list(set(self.ans2normal.values()))
        self.normal_list.sort()
        self.ans2label = {ans:i for i, ans in enumerate(self.normal_list)}
        self.label2ans = {i:ans for i, ans in enumerate(self.normal_list)}
        self.num_answers = len(self.normal_list)  
        if self.split == "train":
            print(f"Answer count: {self.num_answers}") 
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(ann['image_name'])        
        image = Image.open(image_path).convert('RGB')  

        target = torch.zeros(self.num_answers)
        if self.left_right_flip and self.split == "train":
            flip_word_flag = False
            for t in self.transform.transforms:
                if not isinstance(t, transforms.RandomHorizontalFlip):
                    image = t(image)
                elif torch.rand(1) < 0.5:
                    image = F.hflip(image)
                    flip_word_flag = True
            if flip_word_flag:
                for ans in ann['answer']:
                    if flip(pre_answer(ans)) in self.ans2label:
                        target[self.ans2label[flip(pre_answer(ans))]] = 1 
            else:
                for ans in ann['answer']:
                    if pre_answer(ans) in self.ans2label:
                        target[self.ans2label[pre_answer(ans)]] = 1 
        else:
            image = self.transform(image)  
            for ans in ann['answer']:
                if pre_answer(ans) in self.ans2label:
                    target[self.ans2label[pre_answer(ans)]] = 1  

        question = pre_question(ann['question'], self.max_ques_words) # can also use question_rephrase
        question_id = ann['qid']

        return question_id, image, question, target


class vqa_med_2019_cls_dataset(vqa_med_cls_dataset):
    def __init__(self, question_files, transform, eos='[SEP]', split="train", max_ques_words=30, answer_list='', left_right_flip=False):
        self.split = split        
        self.ann = []
        self.id2datum = {}

        idx = 0
        for question_file in question_files:
            with open(question_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split('|')
                    pic_path = os.path.join("/".join(question_file.split('/')[0:-1]), 'images', line[0] + ".jpg")
                    self.ann += [{'image_name':pic_path, 'question':line[-2], 'answer':[line[-1]], 'qid':idx}]
                    idx += 1
                    self.id2datum[idx] = idx

        self.transform = transform
        self.max_ques_words = max_ques_words
        self.eos = eos

        self.left_right_flip = left_right_flip

        self.answer_list = json.load(open(answer_list, 'r'))
        self.ans2normal = {ans:pre_answer(ans) for ans in self.answer_list}
        #flip_list = []
        if self.left_right_flip:
            for ans in self.answer_list:
                pre_ans = pre_answer(ans)
                if flip(pre_ans) != pre_ans and not flip(pre_ans) in self.ans2normal.values():
                    self.ans2normal[flip(pre_ans)] = flip(pre_ans)
                    #flip_list.append(pre_ans)
        #print(flip_list)
        self.normal_list = list(set(self.ans2normal.values()))
        self.normal_list.sort()
        self.ans2label = {ans:i for i, ans in enumerate(self.normal_list)}
        self.label2ans = {i:ans for i, ans in enumerate(self.normal_list)}
        self.num_answers = len(self.normal_list)  
        if self.split == "train":
            print(f"Answer count: {self.num_answers}") 


class slake_cls_dataset(Dataset):
    def __init__(self, question_files, transform, eos='[SEP]', split="train", max_ques_words=30, answer_list='', left_right_flip=False):

        self.split = split        
        self.ann = []
        self.id2qid = {}
        self.qid2id = {}

        idx = 0

        for question_file in question_files:
            with open(question_file, 'r') as f:
                # {"img_id": 384, "img_name": "xmlab384/source.jpg", "question": "What scanning plane does this image belong to?", "answer": "Coronal Plane", "q_lang": "en", "location": "Lung", "modality": "X-Ray", "answer_type": "OPEN", "base_type": "vqa", "content_type": "Plane", "triple": ["vhead", "_", "_"], "qid": 2725}
                lines = json.load(f)
                for line in lines:
                    if line['q_lang'] != "en":
                        continue
                    pic_path = os.path.join("data/slake1.0/imgs", line["img_name"])
                    self.ann += [{'image_name':pic_path, 'question':line["question"], 'answer':line["answer"], 'qid':line["qid"]}]
                    self.id2qid[idx] = int(line["qid"])
                    self.qid2id[int(line["qid"])] = idx
                    idx += 1

        self.transform = transform
        self.max_ques_words = max_ques_words
        self.eos = eos

        self.left_right_flip = left_right_flip

        self.answer_list = json.load(open(answer_list, 'r'))
        self.ans2normal = {ans:pre_answer(ans) for ans in self.answer_list}
        if self.left_right_flip:
            for ans in self.answer_list:
                pre_ans = pre_answer(ans)
                if flip(pre_ans) != pre_ans and not flip(pre_ans) in self.ans2normal.values():
                    self.ans2normal[flip(pre_ans)] = flip(pre_ans)
        self.normal_list = list(set(self.ans2normal.values()))
        self.normal_list.sort()
        self.ans2label = {ans:i for i, ans in enumerate(self.normal_list)}
        self.label2ans = {i:ans for i, ans in enumerate(self.normal_list)}
        self.num_answers = len(self.normal_list)
        if self.split == "train":
            print(f"Answer count: {self.num_answers}")
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(ann['image_name'])        
        image = Image.open(image_path).convert('RGB')  

        target = torch.zeros(self.num_answers)
        if self.left_right_flip and self.split == "train":
            flip_word_flag = False
            for t in self.transform.transforms:
                if not isinstance(t, transforms.RandomHorizontalFlip):
                    image = t(image)
                elif torch.rand(1) < 0.5:
                    image = F.hflip(image)
                    flip_word_flag = True
            if flip_word_flag:
                for ans in [ann['answer']]:
                    if flip(pre_answer(ans)) in self.ans2label:
                        target[self.ans2label[flip(pre_answer(ans))]] = 1 
            else:
                for ans in [ann['answer']]:
                    if pre_answer(ans) in self.ans2label:
                        target[self.ans2label[pre_answer(ans)]] = 1 
        else:
            image = self.transform(image)  
            for ans in [ann['answer']]:
                if pre_answer(ans) in self.ans2label:
                    target[self.ans2label[pre_answer(ans)]] = 1  

        question = pre_question(ann['question'], self.max_ques_words) # can also use question_rephrase
        question_id = ann['qid']

        return question_id, image, question, target


class vqa_rad_cls_dataset(Dataset):
    def __init__(self, question_files, transform, eos='[SEP]', split="train", max_ques_words=30, answer_list='', left_right_flip=False):

        self.split = split        
        self.ann = []
        self.id2qid = {}
        self.qid2id = {}

        idx = 0

        for question_file in question_files:
            with open(question_file, 'r') as f:
                # {"img_id": 384, "img_name": "xmlab384/source.jpg", "question": "What scanning plane does this image belong to?", "answer": "Coronal Plane", "q_lang": "en", "location": "Lung", "modality": "X-Ray", "answer_type": "OPEN", "base_type": "vqa", "content_type": "Plane", "triple": ["vhead", "_", "_"], "qid": 2725}
                lines = json.load(f)
                for line in lines:
                    pic_path = os.path.join("data/vqarad/images", line["image_name"])
                    self.ann += [{'image_name':pic_path, 'question':line["question"], 'answer':line["answer"], 'qid':line["qid"]}]
                    self.id2qid[idx] = int(line["qid"])
                    self.qid2id[int(line["qid"])] = idx
                    idx += 1

        self.transform = transform
        self.max_ques_words = max_ques_words
        self.eos = eos

        self.left_right_flip = left_right_flip

        self.answer_list = json.load(open(answer_list, 'r'))
        self.ans2normal = {ans:pre_answer(ans) for ans in self.answer_list}
        if self.left_right_flip:
            for ans in self.answer_list:
                pre_ans = pre_answer(ans)
                if flip(pre_ans) != pre_ans and not flip(pre_ans) in self.ans2normal.values():
                    self.ans2normal[flip(pre_ans)] = flip(pre_ans)
        self.normal_list = list(set(self.ans2normal.values()))
        self.normal_list.sort()
        self.ans2label = {ans:i for i, ans in enumerate(self.normal_list)}
        self.label2ans = {i:ans for i, ans in enumerate(self.normal_list)}
        self.num_answers = len(self.normal_list)  
        if self.split == "train":
            print(f"Answer count: {self.num_answers}")
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(ann['image_name'])        
        image = Image.open(image_path).convert('RGB')  

        target = torch.zeros(self.num_answers)
        if self.left_right_flip and self.split == "train":
            flip_word_flag = False
            for t in self.transform.transforms:
                if not isinstance(t, transforms.RandomHorizontalFlip):
                    image = t(image)
                elif torch.rand(1) < 0.5:
                    image = F.hflip(image)
                    flip_word_flag = True
            if flip_word_flag:
                for ans in [ann['answer']]:
                    if flip(pre_answer(ans)) in self.ans2label:
                        target[self.ans2label[flip(pre_answer(ans))]] = 1 
            else:
                for ans in [ann['answer']]:
                    if pre_answer(ans) in self.ans2label:
                        target[self.ans2label[pre_answer(ans)]] = 1 
        else:
            image = self.transform(image)  
            for ans in [ann['answer']]:
                if pre_answer(ans) in self.ans2label:
                    target[self.ans2label[pre_answer(ans)]] = 1  

        question = pre_question(ann['question'], self.max_ques_words) # can also use question_rephrase
        question_id = ann['qid']

        return question_id, image, question, target


class retreive_dataset(Dataset):
    def __init__(self, origin_dataset, retrieval_count=3, debug=False, retrieval_by_rank=False):
        self.origin_dataset = origin_dataset
        self.retrieval_count = retrieval_count
        # self.retrieval(model_for_faiss)

        self.retrieval_by_rank_flag = retrieval_by_rank
        self.debug = debug

        # copy
        self.num_answers = self.origin_dataset.num_answers
        self.ann = self.origin_dataset.ann
        self.label2ans =  self.origin_dataset.label2ans

    def retrieval(self, model, text_index, texts, fig_index, figs):
        if self.debug:
            if self.origin_dataset.split == 'train':
                print('Debug mode, no retrieval.') 
            return
        print('Retrieval...')
        self.retrieval_texts = {}
        self.retrieval_images_path = {}
        self.retrieval_prob = {}
        # model, text_index, texts, fig_index, figs = get_faiss_index(model_for_faiss, 
        #                                                             range='pmcp', debug=False)

        imageq = np.array([self.origin_dataset.ann[index]['image_name'] for index in range(len(self.origin_dataset))])
        all_idx, all_sim, ret_texts, ret_figs = retrieval(model, imageq, text_index, fig_index, texts, figs, k=self.retrieval_count)
        for index in range(len(self.origin_dataset)):
            now_idx = []
            now_sim = []
            now_texts = []
            now_figs = []
            for i in range(len(all_idx[index])):
                if not all_idx[index][i] in now_idx:
                    now_idx.append(all_idx[index][i])
                    now_sim.append(all_sim[index][i])
                    now_texts.append(ret_texts[index][i])
                    now_figs.append(ret_figs[index][i])
                else:
                    e = now_idx.index(all_idx[index][i])
                    now_sim[e] = max(now_sim[e], all_sim[index][i])
            self.retrieval_texts[index] = now_texts
            self.retrieval_images_path[index] = now_figs
            self.retrieval_prob[index] = [x / sum(now_sim) for x in now_sim]
        print('Retrieval done.')

    def retrieval_by_prob(self, index):
        ret_text = self.retrieval_texts[index]
        ret_fig_path = self.retrieval_images_path[index]
        ret_prob = self.retrieval_prob[index]
        choose_id = np.random.choice(len(ret_prob), self.retrieval_count, replace=False, p=ret_prob)
        return [ret_text[idx] for idx in choose_id], \
               [self.origin_dataset.transform(Image.open(ret_fig_path[idx.item()]).convert('RGB')) for idx in choose_id]

    def retrieval_by_rank(self, index):
        ret_text = self.retrieval_texts[index]
        ret_fig_path = self.retrieval_images_path[index]
        ret_prob = self.retrieval_prob[index]
        choose_id = torch.topk(torch.FloatTensor(ret_prob), self.retrieval_count)[1]
        return [ret_text[idx.item()] for idx in choose_id], \
               [self.origin_dataset.transform(Image.open(ret_fig_path[idx.item()]).convert('RGB')) for idx in choose_id]

    def __len__(self):
        return self.origin_dataset.__len__()

    def __getitem__(self, index):    
        question_id, image, question, target = self.origin_dataset.__getitem__(index)

        if self.debug:
            ret_text = [question] * self.retrieval_count
            ret_fig = [image] * self.retrieval_count
        else:
            if self.origin_dataset.split == 'train' and not self.retrieval_by_rank_flag:
                ret_text, ret_fig = self.retrieval_by_prob(index)
            else:
                ret_text, ret_fig = self.retrieval_by_rank(index)

        questions = [question] + ret_text
        images = torch.stack([image] + ret_fig, dim=0)

        return question_id, images, questions, target

def vqa_cls_collate_fn(batch):
    id_list, image_list, question_list, target_list = [], [], [], []
    for question_id, image, question, target in batch:
        id_list.append(question_id)
        image_list.append(image)
        question_list.append(question)
        target_list.append(target)
    image_collate = torch.stack(image_list, dim=0)
    return id_list, image_collate, question_list, torch.stack(target_list,dim=0)

def pretrain_collate_fn(batch):
    id_list, image_list, question_list = [], [], []
    for question_id, image, question in batch:
        id_list.append(question_id)
        image_list.append(image)
        question_list.append(question)
    return id_list, torch.stack(image_list,dim=0), question_list

def check_dataset(dataset):
    count = 0
    from tqdm import tqdm
    for data in tqdm(dataset):
        question_id, image, question, target = data
        #import ipdb; ipdb.set_trace()
        if target.sum() == 0:
            count += 1
    print(1 - count / len(dataset))

