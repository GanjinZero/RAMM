import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm
import shutil
import json
# import ipdb
import sys
import numpy as np
from torch.utils.data import DataLoader
from vqa_med_dataset import pretrain_collate_fn
from train_parser import generate_pretrain_parser
from train_utils import generate_pretrain_output_folder_name, generate_pretrain_model, create_transform, configure_optimizers, create_pretrain_dataset
from accelerate import DistributedDataParallelKwargs, Accelerator
try:
    import ruamel.yaml as yaml
except BaseException:
    import ruamel_yaml as yaml

def run(args):
    data_config = yaml.load(open(args.data_config, 'r'), Loader=yaml.Loader)
    model_config = yaml.load(open(args.model_config, 'r'), Loader=yaml.Loader)

    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)    
    
    output_basename = generate_pretrain_output_folder_name(data_config, model_config, args.debug)
    accelerator.print(output_basename)
    output_path = os.path.join(args.output_base_dir, output_basename)
    
    try:
        world_size = torch.distributed.get_world_size()
    except:
        world_size = 1
    if world_size > 1:
        output_path = output_path + f"_{world_size}gpu"

    try:
        os.system(f"mkdir -p {output_path}")
    except BaseException:
        pass

    shutil.copyfile(args.data_config, os.path.join(output_path, "data_config.yml"))
    shutil.copyfile(args.model_config, os.path.join(output_path, "model_config.yml"))
    
    # no dev
    train_transform, _ = create_transform(model_config)
    train_dataset = create_pretrain_dataset(data_config, train_transform)

    model = generate_pretrain_model(model_config, train_dataset, args.debug).to(accelerator.device)
    if model_config.get('ckpt', ''):
        # init from mplug/albef-style checkpoint
        checkpoint = torch.load(model_config['ckpt'], map_location='cpu')
        try:
            try:
                state_dict = checkpoint['model']
            except:
                state_dict = checkpoint['module']
        except:
            state_dict = checkpoint.state_dict()

        msg = model.load_state_dict(state_dict, strict=False)
        missing_key = [x for x in msg[0] if x.find('cross') == -1]
        print('load checkpoint from %s' % model_config['ckpt'])
        print(missing_key)

    train_dataloader = DataLoader(train_dataset, batch_size=model_config['batch_size_train'], collate_fn=pretrain_collate_fn, shuffle=True, num_workers=8, pin_memory=True)    
    optimizer, scheduler_step = configure_optimizers(model, train_dataloader, model_config)

    model, optimizer, train_dataloader = \
        accelerator.prepare(model, optimizer, train_dataloader)

    steps = 0

    for epoch_idx in range(1, model_config['train_epoch'] + 1):
        steps, average_epoch_loss = \
             train_one_epoch(model, steps, train_dataloader, optimizer, scheduler_step, model_config, accelerator)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            # torch.save(model, os.path.join(output_path, f"epoch{epoch_idx}.pth"))
            accelerator.save(accelerator.unwrap_model(model), os.path.join(output_path, f"epoch{epoch_idx}.pth"))
            print_metrics(average_epoch_loss, 'Train_Epoch' + str(epoch_idx), os.path.join(output_path, 'metric_log'))

def train_one_epoch(model, steps, train_dataloader, optimizer, scheduler, model_config, accelerator=None):
    model.train()
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", ascii=True, disable=not accelerator.is_local_main_process)

    epoch_loss = 0.0
    epoch_mlm = 0.0
    epoch_ita = 0.0
    epoch_itm = 0.0
    for batch_idx, batch in enumerate(epoch_iterator):
        images = batch[1].to(accelerator.device)
        texts = batch[2]
        
        if steps >= len(train_dataloader):
            alpha = model_config['alpha']
        else:
            alpha = model_config['alpha']*min(1,steps/len(train_dataloader)) 
        loss_mlm, loss_ita, loss_itm = model(images, texts, alpha=alpha)
        loss = loss_mlm + loss_ita + loss_itm

        batch_loss = float(loss.item())
        epoch_loss += batch_loss
        epoch_mlm += float(loss_mlm.item())
        epoch_ita += float(loss_ita.item())
        epoch_itm += float(loss_itm.item())

        if model_config['gradient_accumulation_steps'] > 1:
            loss = loss / model_config['gradient_accumulation_steps']

        # loss.backward()
        accelerator.backward(loss)
        epoch_iterator.set_description("E: %0.4f, B: %0.4f, E_mlm: %0.4f, B_mlm: %0.4f, E_ita: %0.4f, B_ita: %0.4f, E_itm: %0.4f, B_itm: %0.4f" % (epoch_loss / (batch_idx + 1), batch_loss, epoch_mlm / (batch_idx + 1), loss_mlm, epoch_ita / (batch_idx + 1), loss_ita, epoch_itm / (batch_idx + 1), loss_itm))

        if (steps + 1) % model_config['gradient_accumulation_steps'] == 0:
            accelerator.clip_grad_norm_(
                 model.parameters(), model_config['max_grad_norm'])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            model.zero_grad()

        steps += 1

    average_epoch_loss = {'loss': epoch_loss / (batch_idx + 1),
                          'mlm': epoch_mlm / (batch_idx + 1),
                          'ita': epoch_ita / (batch_idx + 1),
                          'itm': epoch_itm / (batch_idx + 1)}

    return steps, average_epoch_loss

def print_metrics(metric, tag='', output_path=None):
    if output_path is None:
        print(tag, metric)
    else:
        with open(output_path, 'a+') as f:
            f.write(tag + ":" + str(metric) + "\n")

def main():
    parser = generate_pretrain_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
