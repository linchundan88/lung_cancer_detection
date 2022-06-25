import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1')
parser.add_argument('--task_type', default='ACDC2019_v2')
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
from libs.neuralNetworks.semanticSegmentation.models.my_get_model_helper import get_model
from libs.neuralNetworks.my_dataset import Dataset_CSV_SEM_SEG
import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from segmentation_models_pytorch.losses.dice import DiceLoss
from libs.neuralNetworks.semanticSegmentation.metrics.metrics_numpy import get_confusion_matrix, get_metrics #CPU, slow




path_models = Path(__file__).resolve().parent / 'trained_models' / 'v2' / 'dice'
model_file1 = path_models / 'Unet_resnet34_valid_loss_0.325_epoch3.pth'
model1 = get_model(model_type='Unet', encoder_name='resnet34', model_file=model_file1)

image_shape = (512, 512)
mask_threshold = 127

path_csv = Path(__file__).resolve().parent / 'datafiles'
csv_valid = path_csv / f'{args.task_type}_test.csv'
ds_valid = Dataset_CSV_SEM_SEG(csv_file=csv_valid, image_shape=image_shape, mask_threshold=mask_threshold)
dataloader_valid = DataLoader(ds_valid, batch_size=32, num_workers=8, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 0:
    model1.to(device)
if torch.cuda.device_count() > 1:
    model1 = nn.DataParallel(model1)

criterion = DiceLoss(mode='binary', from_logits=True)  # segmentation_models_pytorch loss function

epoch_valid_loss_valid, epoch_tp_valid, epoch_tn_valid, epoch_fp_valid, epoch_fn_valid = 0, 0, 0, 0, 0
with torch.inference_mode():  #better than torch.no_grad  using pytorch>1.9 disabling view tracking and version counter bumps.”
    for batch_idx, (images, masks) in enumerate(dataloader_valid):
        print(f'batch:{batch_idx}')
        images = images.to(device)
        masks = masks.to(device, dtype=torch.float32)
        outputs = model1(images)
        loss = criterion(outputs, masks)

        outputs = torch.sigmoid(outputs)

        outputs = outputs.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()

        epoch_valid_loss_valid += loss.item()
        tp, tn, fp, fn = get_confusion_matrix(outputs, masks, threshold=0.5)
        epoch_tp_valid += tp
        epoch_tn_valid += tn
        epoch_fp_valid += fp
        epoch_fn_valid += fn

    epoch_valid_loss_valid /= (batch_idx + 1)
    epoch_acc_valid, epoch_sen_valid, epoch_spe_valid, epoch_iou_valid, epoch_dice_valid =\
        get_metrics(epoch_tp_valid, epoch_tn_valid, epoch_fp_valid, epoch_fn_valid)

    print(f'losses:{epoch_valid_loss_valid:8.2f}')
    print(f'acc:{epoch_acc_valid:5.3f}, sen:{epoch_sen_valid:5.3f}, spe:{epoch_spe_valid:5.3f}')
    print(f'iou:{epoch_iou_valid:5.3f}, dice:{epoch_dice_valid:5.3f}')

print('OK')