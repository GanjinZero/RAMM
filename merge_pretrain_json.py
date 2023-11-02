import json
from tqdm import tqdm
from PIL import Image


def merge(file_list, output_path):
    opt = []
    for file in file_list:
        print(file)
        with open(file, 'r') as f:
            lines = json.load(f)
        for line in tqdm(lines):
            try:
                Image.open(line["image"]).convert('RGB')
                if isinstance(line['caption'], list):
                    question = line['caption'][0]
                else:
                    question = line["caption"]
                opt += [{'image_name':line["image"], 'caption':question}]
            except:
                pass
    with open(output_path, 'w') as f:
        json.dump(opt, f, indent=4)
      