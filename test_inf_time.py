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
from vqa_med_dataset import vqa_cls_collate_fn
from train_parser import generate_parser
from train_utils import generate_output_folder_name, generate_model, create_transform, configure_optimizers, create_dataset
from accelerate import DistributedDataParallelKwargs, Accelerator
from model.losses import LabelSmoothingCrossEntropy
try:
    import ruamel.yaml as yaml
except BaseException:
    import ruamel_yaml as yaml
from model.ema import ExponentialMovingAverage
# from model.swin_helper import swin_adapt_position_encoding

def run(args):
    data_config = yaml.load(open(args.data_config, 'r'), Loader=yaml.Loader)
    model_config = yaml.load(open(args.model_config, 'r'), Loader=yaml.Loader)
    model_config['tag'] = args.tag

    data_config["left_right_flip"] = model_config.get("left_right_flip", False)
    data_config["retrieval"] = model_config.get("retrieval", False)
    data_config["retrieval_count"] = model_config.get("retrieval_count", 1)
    # data_config["faiss_model"] = "./pretrain_outputs/pretrain_pmcp_swin_base_patch4_window7_224_in22k_albef_30epoch_224res_meter6_pubmedbert_lr5e-05_tlr5e-05_xlr0.0002_clr0.0002_8gpu/epoch30.pth" # hard code
    # data_config["faiss_model"] = "./pretrain_outputs/pretrain_fuse_swin_base_patch4_window7_224_in22k_albef_30epoch_224resbsz64_meter6_pubmedbert_lr1e-05_tlr1e-05_xlr5e-05_clr5e-05_8gpu/epoch30.pth" # hard code
    data_config["faiss_model"] = model_config.get("ckpt", "")
    data_config['retrieval_range'] = model_config.get("retrieval_range", 'fuse')
    data_config['retrieval_by_rank'] = model_config.get("retrieval_by_rank", False)
    #print(data_config)
    
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)    
    
    output_basename = generate_output_folder_name(data_config, model_config, args.debug)
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
    train_transform, test_transform = create_transform(model_config)
    train_dataset, dev_dataset, test_dataset = create_dataset(data_config, train_transform, test_transform, args.debug)

    # trunc dataset
    train_dataset = train_dataset[0:64]
    dev_dataset = dev_dataset[0:10]
    test_dataset = test_dataset[0:10]

    model = generate_model(model_config, train_dataset, args.debug).to(accelerator.device)
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
    
        # if model_config["backbone"].find("16") >= 0:
        #     num_patches = int(model_config["image_res"] * model_config["image_res"]/(16*16))
        # else:
        if model_config["backbone"].find("mplug") >= 0:
            num_patches = int(model_config["image_res"] * model_config["image_res"]/(14*14)) # hard code for 14
            pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float()) # hard code for base-size model

            from vit import resize_pos_embed
            pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0),
                                        pos_embed.unsqueeze(0))
            state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

            for key in list(state_dict.keys()):
                if 'bert' in key:
                    new_key = key.replace('bert.', '')
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
                if key.startswith('visual_encoder.visual'):
                    new_key = key.replace('visual_encoder.visual', 'visual_encoder')
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        if model_config["backbone"].startswith('swin') and model_config["image_res"] != 224:
            # delete relative_position_index since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
            for k in relative_position_index_keys:
                del state_dict[k]

            # delete relative_coords_table since we always re-init it
            relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
            for k in relative_position_index_keys:
                del state_dict[k]

            # delete attn_mask since we always re-init it
            attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
            for k in attn_mask_keys:
                del state_dict[k]

            # bicubic interpolate relative_position_bias_table if not match
            relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k and k.find("_m") == -1]
            for k in relative_position_bias_table_keys:
                relative_position_bias_table_pretrained = state_dict[k]
                relative_position_bias_table_current = model.state_dict()[k]
                L1, nH1 = relative_position_bias_table_pretrained.size()
                L2, nH2 = relative_position_bias_table_current.size()
                if nH1 != nH2:
                    print(f"Error in loading {k}, passing......")
                else:
                    if L1 != L2:
                        # bicubic interpolate relative_position_bias_table if not match
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                            relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                            mode='bicubic')
                        state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

            # bicubic interpolate absolute_pos_embed if not match
            absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k and k.find("_m") == -1]
            for k in absolute_pos_embed_keys:
                # dpe
                absolute_pos_embed_pretrained = state_dict[k]
                absolute_pos_embed_current = model.state_dict()[k]
                _, L1, C1 = absolute_pos_embed_pretrained.size()
                _, L2, C2 = absolute_pos_embed_current.size()
                if C1 != C2:
                    print(f"Error in loading {k}, passing......")
                else:
                    if L1 != L2:
                        S1 = int(L1 ** 0.5)
                        S2 = int(L2 ** 0.5)
                        absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                        absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                        absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                            absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                        absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                        absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                        state_dict[k] = absolute_pos_embed_pretrained_resized

        msg = model.load_state_dict(state_dict, strict=False)
        missing_key = [x for x in msg[0] if x.find('cross') == -1]
        print('load checkpoint from %s' % model_config['ckpt'])
        print(missing_key)
        #import ipdb; ipdb.set_trace()

    if 'ema' in model_config:
        ema_value = model_config['ema']
        print(f'EMA {ema_value}')
        ema = ExponentialMovingAverage(model.parameters(), model_config['ema'])
    else:
        ema = None

    train_dataloader = DataLoader(train_dataset, batch_size=model_config['batch_size_train'], collate_fn=vqa_cls_collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    if dev_dataset is not None:
        dev_dataloader = DataLoader(dev_dataset, batch_size=model_config['batch_size_train'], collate_fn=vqa_cls_collate_fn, shuffle=True, num_workers=8, pin_memory=True)
    else:
        dev_dataloader = None
    test_dataloader = DataLoader(test_dataset, batch_size=model_config['batch_size_test'], collate_fn=vqa_cls_collate_fn, shuffle=False, num_workers=8, pin_memory=True)
    
    optimizer, scheduler_step = configure_optimizers(model, train_dataloader, model_config)

    model, optimizer, train_dataloader = \
        accelerator.prepare(model, optimizer, train_dataloader)

    if accelerator.is_local_main_process and args.debug:
        test_metric, test_predict = eval_func(model, test_dataloader, accelerator.device, True, ema)
        save_predict(test_predict, os.path.join(output_path, 'debug_predict.txt'), test_dataset)
        print_metrics(test_metric, 'DEBUG')

    steps = 0
    best_dev_metric = {}
    best_test_metric = {}
    best_epoch_idx = 0

    for epoch_idx in range(1, model_config['train_epoch'] + 1):
        epoch_test_metric, steps, epoch_test_predict, epoch_dev_metric, epoch_dev_predict, average_epoch_loss = \
             train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler_step, model_config, accelerator, ema)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            # torch.save(model, os.path.join(output_path, f"epoch{epoch_idx}.pth"))
            if ema:
                ema.store()
                ema.copy_to()
            accelerator.save(accelerator.unwrap_model(model), os.path.join(output_path, f"epoch{epoch_idx}.pth"))
            if ema:
                ema.restore()
            print_metrics(average_epoch_loss, 'Train_Epoch' + str(epoch_idx), os.path.join(output_path, 'metric_log'))
            if epoch_dev_metric is not None:
                print_metrics(epoch_dev_metric, 'Dev_Epoch' + str(epoch_idx))
                print_metrics(epoch_dev_metric, 'Dev_Epoch' + str(epoch_idx), os.path.join(output_path, 'metric_log'))
            print_metrics(epoch_test_metric, 'Test_Epoch' + str(epoch_idx))
            print_metrics(epoch_test_metric, 'Test_Epoch' + str(epoch_idx), os.path.join(output_path, 'metric_log'))
            # TODO: save epoch_test_predict
            if epoch_dev_predict is not None:
                save_predict(epoch_dev_predict, os.path.join(output_path, f'dev_{epoch_idx}.txt'), dev_dataset)
            save_predict(epoch_test_predict, os.path.join(output_path, f'test_{epoch_idx}.txt'), test_dataset)

        if not best_dev_metric:
            best_dev_metric = epoch_dev_metric
            best_test_metric = epoch_test_metric
            best_epoch_idx = epoch_idx
        else:
            if epoch_dev_metric is not None:
                if epoch_dev_metric['acc'] >= best_dev_metric['acc']:
                    best_dev_metric = epoch_dev_metric
                    best_test_metric = epoch_test_metric
                    best_epoch_idx = epoch_idx
            else:
                # if no dev, use last epoch as best
                best_dev_metric = epoch_dev_metric
                best_test_metric = epoch_test_metric
                best_epoch_idx = epoch_idx

    if accelerator.is_local_main_process:
        best_train_metric, best_train_predict = eval_func(model, train_dataloader, accelerator.device, True, ema)
        save_predict(best_train_predict, os.path.join(output_path, f'train_best.txt'), train_dataset)
        print_metrics(best_train_metric, 'Best_Train_Epoch' + str(best_epoch_idx))
        print_metrics(best_train_metric, 'Best_Train_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        
        if best_dev_metric:
            print_metrics(best_dev_metric, 'Best_Dev_Epoch' + str(best_epoch_idx))
            print_metrics(best_dev_metric, 'Best_Dev_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
            best_dev_predict = os.path.join(output_path, f'dev_{epoch_idx}.txt')
            new_dev_predict = os.path.join(output_path, f'dev_best.txt')
            os.system(f'cp {best_dev_predict} {new_dev_predict}')
        
        print_metrics(best_test_metric, 'Best_Test_Epoch' + str(best_epoch_idx))
        print_metrics(best_test_metric, 'Best_Test_Epoch' + str(best_epoch_idx), os.path.join(output_path, 'metric_log'))
        best_test_predict = os.path.join(output_path, f'test_{epoch_idx}.txt')
        new_test_predict = os.path.join(output_path, f'test_best.txt')
        os.system(f'cp {best_test_predict} {new_test_predict}')

        best_path = os.path.join(output_path, f"epoch{best_epoch_idx}.pth")
        new_path = os.path.join(output_path, "best_epoch.pth")
        os.system(f'cp {best_path} {new_path}')


def train_one_epoch(model, steps, train_dataloader, dev_dataloader, test_dataloader, optimizer, scheduler, model_config, accelerator=None, ema=None):
    model.train()
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", ascii=True, disable=not accelerator.is_local_main_process)

    epoch_loss = 0.0
    for batch_idx, batch in enumerate(epoch_iterator):
        #batch_gpu = tuple([x.to(accelerator.device) for x in batch])
        images = batch[1].to(accelerator.device)
        labels = batch[-1].to(accelerator.device)

        if not hasattr(model, 'text_encoder'):
            logits = model(images)
        else:
            texts = batch[2]
            logits = model(images, texts)

        if not 'loss' in model_config or model_config['loss'] == 'ce':
            loss = - torch.sum(F.log_softmax(logits, dim=1) * labels, dim=1).mean()
        elif model_config['loss'] == 'label_smooth':
            loss = LabelSmoothingCrossEntropy()(logits, labels)

        if 'rdrop' in model_config and model_config['rdrop'] > 0:
            logits2 = model(images, texts)
            logits2_loss = - torch.sum(F.log_softmax(logits2, dim=1) * labels, dim=1).mean()
            p_loss = F.kl_div(F.log_softmax(logits, dim=-1), F.softmax(logits2, dim=-1), reduction='none')
            q_loss = F.kl_div(F.log_softmax(logits2, dim=-1), F.softmax(logits, dim=-1), reduction='none')
            rdrop_loss = (p_loss.mean() + q_loss.mean()) / 2
            loss = (loss + logits2_loss) / 2 + rdrop_loss * model_config['rdrop']

        batch_loss = float(loss.item())
        epoch_loss += batch_loss

        if model_config['gradient_accumulation_steps'] > 1:
            loss = loss / model_config['gradient_accumulation_steps']

        # loss.backward()
        accelerator.backward(loss)
        epoch_iterator.set_description("Epoch: %0.4f, Batch: %0.4f" % (epoch_loss / (batch_idx + 1), batch_loss))

        if (steps + 1) % model_config['gradient_accumulation_steps'] == 0:
            accelerator.clip_grad_norm_(
                 model.parameters(), model_config['max_grad_norm'])
            optimizer.step()
            if scheduler is not None:
                scheduler.step()  # Update learning rate schedule
            if ema:
                ema.update()
            model.zero_grad()

        steps += 1

    average_epoch_loss = {'loss':epoch_loss / (batch_idx + 1)}

    tqdm_bar = False
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        if dev_dataloader is not None:
            dev_metric, dev_predict = eval_func(model, dev_dataloader, accelerator.device, tqdm_bar, ema)
        else:
            dev_metric, dev_predict = None, None
        test_metric, test_predict = eval_func(model, test_dataloader, accelerator.device, tqdm_bar, ema)
    else:
        test_metric = None
        test_predict = None
        dev_metric = None
        dev_predict = None
    return test_metric, steps, test_predict, dev_metric, dev_predict, average_epoch_loss

def eval_func(model, dataloader, device, tqdm_bar, ema=None):
    model.eval()
    if ema:
        ema.store()
        ema.copy_to()

    all_outputs = []
    all_labels = []
    # device = args.device if args is not None else device
    it = tqdm(dataloader) if tqdm_bar else dataloader
    with torch.no_grad():
        for batch in it:
            images = batch[1].to(device)
            labels = batch[-1].to(device)
            if isinstance(model, DistributedDataParallel):
                if not hasattr(model.module, 'text_encoder'):
                    logits = model.module(images)
                else:
                    texts = batch[2]
                    logits = model.module(images, texts)
            else:
                if not hasattr(model, 'text_encoder'):
                    logits = model(images)
                else:
                    texts = batch[2]
                    logits = model(images, texts)

            all_outputs.append(logits.cpu().detach())
            all_labels.append(labels.cpu().detach())

    if ema:
        ema.restore()
            
    all_outputs = torch.cat(all_outputs, dim=0) # batch * class
    all_labels = torch.cat(all_labels, dim=0).long()

    all_pred = torch.argmax(all_outputs, dim=1) # batch
    correct = 0
    for i in range(all_labels.shape[0]):
        if all_labels[i][all_pred[i]] == 1:
            correct += 1

    return {'acc': correct / all_labels.shape[0]}, all_outputs

def print_metrics(metric, tag='', output_path=None):
    if output_path is None:
        print(tag, metric)
    else:
        with open(output_path, 'a+') as f:
            f.write(tag + ":" + str(metric) + "\n")

def save_predict(predict, output_path, dataset):
    all_pred = torch.argmax(predict, dim=1).cpu().detach().numpy().tolist() # batch
    opt = []
    for ind, pred in enumerate(all_pred):
        ann = dataset.ann[ind]
        question_id = ann['qid']
        answer = ann["answer"]
        opt.append({'qid':question_id, 'predict':dataset.label2ans[pred], 'answer':answer})
    with open(output_path, 'w') as f:
        json.dump(opt, f, indent=4)

def main():
    parser = generate_parser()
    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()

