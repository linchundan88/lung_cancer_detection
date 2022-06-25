import os
from pathlib import Path
import numpy as np
import torch
import tqdm
from torch import distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader
from libs.neuralNetworks.semanticSegmentation.metrics.metrics_numpy import get_confusion_matrix, get_metrics
import logging
from datetime import datetime
from tqdm import tqdm

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, config):
    print(f'rank:{rank}')

    setup(rank, world_size)
    config.model.cuda(rank)
    # model.to(rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(config.model)
    ddp_model = DDP(model, device_ids=[rank])
    sampler_train = DistributedSampler(config.ds_train, world_size, rank)  # default_value dist.get_rank()
    loader_train = DataLoader(config.ds_train, batch_size=config.batch_size, sampler=sampler_train,
                              num_workers=config.num_workers, pin_memory=True)
    if config.save_model_dir:
        config.save_model_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(filename=config.save_model_dir / f'train{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}.log', level=logging.DEBUG)

    if config.amp:
        scaler = GradScaler()

    for epoch in range(config.epochs_num):
        sampler_train.set_epoch(epoch)
        print(f'training epoch {epoch}/{5 - 1} on rank {rank} at:' + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")) #train_object['epochs_num']
        ddp_model.train()

        epoch_loss_train, epoch_tp_train, epoch_tn_train, epoch_fp_train, epoch_fn_train = 0, 0, 0, 0, 0

        for batch_idx, (images, masks) in enumerate(tqdm(loader_train)):
        # for batch_idx, (images, masks) in enumerate(loader_train):  # show training batch details
            images = images.to(rank)
            masks = masks.to(rank, dtype=torch.float32)

            if (config.accumulate_grads_times is None) or (config.accumulate_grads_times is not None and batch_idx % config.accumulate_grads_times == 0):
                config.optimizer.zero_grad()
            with autocast(enabled=config.amp):
                outputs = ddp_model(images)
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

            epoch_loss_train += loss.item()  # loss function setting reduction='mean'
            tp, tn, fp, fn = get_confusion_matrix(outputs, masks, threshold=0.5)
            epoch_tp_train += tp
            epoch_tn_train += tn
            epoch_fp_train += fp
            epoch_fn_train += fn

            if config.save_model_dir:  #logging
                logging.info(f'epoch:{epoch} training batch:{batch_idx}, losses:{loss.item():8.2f}')

        epoch_loss_train /= (batch_idx + 1)
        epoch_acc_train, epoch_sen_train, epoch_spe_train, epoch_iou_train, epoch_dice_train = get_metrics(epoch_tp_train, epoch_tn_train, epoch_fp_train, epoch_fn_train)

        print(f'training epoch{epoch} rank:{rank} metrics:')
        print(f'rank:{rank}  losses:{epoch_loss_train:8.2f}')
        print(f'rank:{rank} acc:{epoch_acc_train:5.3f}, sen:{epoch_sen_train:5.3f}, spe:{epoch_spe_train:5.3f}')
        print(f'rank:{rank} iou:{epoch_iou_train:5.3f}, dice:{epoch_dice_train:5.3f}')

        config.scheduler.step()

        for index, ds_valid in enumerate(config.list_ds_valid):
            print(f'compute validation rank:{rank} dataset index：{index}...')
            ddp_model.eval()
            epoch_loss_valid, epoch_tp_valid, epoch_tn_valid, epoch_fp_valid, epoch_fn_valid = 0, 0, 0, 0, 0
            sampler_valid = DistributedSampler(ds_valid, world_size, rank)
            dataloader_valid = DataLoader(ds_valid, sampler=sampler_valid, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)

            with torch.inference_mode():  #better than torch.no_grad  using pytorch>1.9 disabling view tracking and version counter bumps.”
                for batch_idx, (images, masks) in enumerate(tqdm(dataloader_valid)):
                    images = images.to(rank)
                    masks = masks.to(rank, dtype=torch.float32)
                    outputs = ddp_model(images)
                    loss = config.criterion(outputs, masks)

                    if config.activation == 'sigmoid':
                        outputs = torch.sigmoid(outputs)

                    outputs = outputs.detach().cpu().numpy()
                    masks = masks.detach().cpu().numpy()

                    epoch_loss_valid += loss.item()
                    tp, tn, fp, fn = get_confusion_matrix(outputs, masks, threshold=0.5)
                    epoch_tp_valid += tp
                    epoch_tn_valid += tn
                    epoch_fp_valid += fp
                    epoch_fn_valid += fn

                    if config.save_model_dir:  # logging
                        logging.info(f'epoch:{epoch} validation batch:{batch_idx}, losses:{loss.item():8.2f}')

            epoch_loss_valid /= (batch_idx + 1)

            list_epoch_loss_valid = [epoch_loss_valid]
            result_array = np.array(epoch_loss_valid, epoch_tp_valid, epoch_tn_valid, epoch_fp_valid, epoch_fn_valid)
            tensor_result = torch.from_numpy(result_array)
            if rank != 0:
                dist.send(tensor=tensor_result, dst=0)
            else:
                for process1 in world_size:
                    tensor_receive = torch.zeros(5)
                    dist.recv(tensor=tensor_receive, src=process1)
                    receive_resuilts = tensor_receive.cpu().numpy()

                    list_epoch_loss_valid.append(receive_resuilts[0])
                    epoch_tp_valid += receive_resuilts[1]
                    epoch_tn_valid += receive_resuilts[2]
                    epoch_fp_valid += receive_resuilts[3]
                    epoch_fn_valid += receive_resuilts[4]

            if rank == 0:
                epoch_loss_valid = np.mean(np.array(list_epoch_loss_valid))
                epoch_acc_valid, epoch_sen_valid, epoch_spe_valid, epoch_iou_valid, epoch_dice_valid = get_metrics(epoch_tp_valid, epoch_tn_valid, epoch_fp_valid, epoch_fn_valid)

                print(f'validation epoch{epoch},  metrics:')
                print(f'losses:{epoch_loss_valid:8.2f}')
                print(f'acc:{epoch_acc_valid:5.3f}, sen:{epoch_sen_valid:5.3f}, spe:{epoch_spe_valid:5.3f}')
                print(f'iou:{epoch_iou_valid:5.3f}, dice:{epoch_dice_valid:5.3f}')

        if config.save_model_dir and rank == 0:
            save_model_file = Path(config.save_model_dir) / f'valid_loss_{round(epoch_loss_valid, 3)}_epoch{epoch}.pth'
            print('save model:', save_model_file)
            torch.save(ddp_model.state_dict(), save_model_file)

    cleanup()