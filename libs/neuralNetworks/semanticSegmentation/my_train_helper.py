'''
  data parallel
'''

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from libs.neuralNetworks.semanticSegmentation.metrics.metrics_numpy import get_confusion_matrix, get_metrics #CPU, slow
from tqdm import tqdm
import logging
from datetime import datetime


def my_train(config):
    assert config.activation in [None, 'sigmoid'], f'activation function {config.activation} error!' \

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0:
        config.model.to(device)
    if torch.cuda.device_count() > 1:
        config.model = nn.DataParallel(config.model)
    if config.amp:
        scaler = GradScaler()

    loader_train = DataLoader(config.ds_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    if config.save_model_dir:
        config.save_model_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=config.save_model_dir / f'train{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log', level=logging.DEBUG)

    for epoch in range(config.epochs_num):
        print(f'training epoch {epoch}/{config.epochs_num - 1} at:' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        config.model.train()

        epoch_loss, epoch_tp, epoch_tn, epoch_fp, epoch_fn = 0, 0, 0, 0, 0

        for batch_idx, (images, masks) in enumerate(tqdm(loader_train)):
        # for batch_idx, (images, masks) in enumerate(loader_train):  # show training batch details
            images = images.to(device)
            masks = masks.to(device, dtype=torch.float32)

            if (config.accumulate_grads_times is None) or (config.accumulate_grads_times is not None and batch_idx % config.accumulate_grads_times == 0):
                config.optimizer.zero_grad()
            with autocast(enabled=config.amp):
                outputs = config.model(images)
                loss = config.criterion(outputs, masks)
            if (config.accumulate_grads_times is None) or (config.accumulate_grads_times is not None and batch_idx % config.accumulate_grads_times == 0):
                if not config.amp:
                    loss.backward()
                    config.optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(config.optimizer)
                    scaler.update()
            else:  # do not update parameters
                if not config.amp:
                    loss.backward()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(config.optimizer)

            epoch_loss += loss.item()  # loss function setting reduction='mean'
            tp, tn, fp, fn = get_confusion_matrix(outputs, masks, threshold=0.5)
            epoch_tp += tp
            epoch_tn += tn
            epoch_fp += fp
            epoch_fn += fn

            if config.save_model_dir:  #logging
                logging.info(f'epoch:{epoch} training batch:{batch_idx}, losses:{loss.item():8.2f}')

        epoch_loss /= (batch_idx + 1)
        epoch_acc, epoch_sen, epoch_spe, epoch_iou, epoch_dice = get_metrics(epoch_tp, epoch_tn, epoch_fp, epoch_fn)

        print(f'training epoch{epoch} metrics:')
        print(f'losses:{epoch_loss:8.2f}')
        print(f'acc:{epoch_acc:5.3f}, sen:{epoch_sen:5.3f}, spe:{epoch_spe:5.3f}')
        print(f'iou:{epoch_iou:5.3f}, dice:{epoch_dice:5.3f}')

        config.scheduler.step()

        for index, ds_valid in enumerate(config.list_ds_valid):
            print(f'epoch:{epoch} compute validation dataset index：{index}...')
            loss_validate = my_validate(ds_valid, config)

        if config.save_model_dir:
            save_model_file = config.save_model_dir / f'valid_loss_{round(loss_validate, 3)}_epoch{epoch}.pth'
            try:
                state_dict = config.model.module.state_dict()
            except AttributeError:
                state_dict = config.model.state_dict()
            print('save model:', save_model_file)
            torch.save(state_dict, save_model_file)



def my_validate(ds_valid, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.model.eval()

    epoch_valid_loss_valid, epoch_tp_valid, epoch_tn_valid, epoch_fp_valid, epoch_fn_valid = 0, 0, 0, 0, 0
    dataloader_valid = DataLoader(ds_valid, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)

    with torch.inference_mode():  #better than torch.no_grad  using pytorch>1.9 disabling view tracking and version counter bumps.”
        for batch_idx, (images, masks) in enumerate(tqdm(dataloader_valid)):
            images = images.to(device)
            masks = masks.to(device, dtype=torch.float32)
            outputs = config.model(images)
            loss = config.criterion(outputs, masks)

            if config.activation == 'sigmoid':
                outputs = torch.sigmoid(outputs)

            outputs = outputs.detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()

            epoch_valid_loss_valid += loss.item()
            tp, tn, fp, fn = get_confusion_matrix(outputs, masks, threshold=0.5)
            epoch_tp_valid += tp
            epoch_tn_valid += tn
            epoch_fp_valid += fp
            epoch_fn_valid += fn

            if config.save_model_dir:  # logging
                logging.info(f'validation batch:{batch_idx}, losses:{loss.item():8.2f}')


    epoch_valid_loss_valid /= (batch_idx + 1)
    epoch_acc_valid, epoch_sen_valid, epoch_spe_valid, epoch_iou_valid, epoch_dice_valid =\
        get_metrics(epoch_tp_valid, epoch_tn_valid, epoch_fp_valid, epoch_fn_valid)

    print(f'validation metrics:')
    print(f'losses:{epoch_valid_loss_valid:8.2f}')
    print(f'acc:{epoch_acc_valid:5.3f}, sen:{epoch_sen_valid:5.3f}, spe:{epoch_spe_valid:5.3f}')
    print(f'iou:{epoch_iou_valid:5.3f}, dice:{epoch_dice_valid:5.3f}')

    return epoch_valid_loss_valid


