import os
import json
from PIL import Image
from torch.utils.data import Dataset
import re
from tqdm import tqdm

class pretrain_dataset(Dataset):
    def __init__(self, question_files, transform):
        self.ann = []
        for question_file in question_files:
            print(question_file)
            cnt = 0
            with open(question_file, 'r') as f:
                lines = json.load(f)
            for line in tqdm(lines):
                #if len(self.ann) > 4000:
                #    break 
                if os.path.exists(line["image"]):
                    try:
                        Image.open(line["image"]).convert('RGB') # check broke image
                        if 'image_id' in line:
                            qid = line["image_id"]
                        else:
                            qid = 0
                        if isinstance(line['caption'], list):
                            question = line['caption'][0]
                        else:
                            question = line["caption"]
                        self.ann += [{'image_name':line["image"], 'question':question, 'qid':qid}]
                        cnt += 1
                        #print([{'image_name':line["image"], 'question':question, 'qid':qid}])
                    except:
                        #print(line)
                        pass
            print(cnt)

        self.transform = transform
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        ann = self.ann[index]
        
        image_path = os.path.join(ann['image_name'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        question = ann['question']
        
        question_id = ann['qid']
        return question_id, image, question

