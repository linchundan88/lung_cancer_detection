import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1')
parser.add_argument('--task_type', default='ACDC2019_v1')
parser.add_argument('--csv_file', default='ACDC2019_v1_test')
parser.add_argument('--path_output', default='/disk_data/data/ACDC_2019/Training/predict_results_2022_6_27/')
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
import pandas as pd
import cv2
import numpy as np
from libs.neuralNetworks.semanticSegmentation.models.my_get_model_helper import get_model
from libs.neuralNetworks.my_dataset import Dataset_CSV_SEM_SEG
from torch.utils.data import DataLoader
from libs.neuralNetworks.semanticSegmentation.metrics.metrics_numpy import get_confusion_matrix, get_metrics #CPU, slow
from libs.neuralNetworks.semanticSegmentation.my_predict_helper import predict_multi_models


#region define models
image_shape = (512, 512)
model_dicts = []

if args.task_type == 'ACDC2019_v1':
    path_models = Path(__file__).resolve().parent / 'trained_models' / 'v1' / 'dice'

    model_file1 = path_models / 'Unet_densenet121_valid_loss_0.319_epoch3.pth'
    model1 = get_model(model_type='Unet', encoder_name='densenet121', model_file=model_file1)
    model_dict1 = dict(model=model1, model_weight=1, image_shape=image_shape, activation='sigmoid', batch_size=32)
    model_dicts.append(model_dict1)

    model_file1 = path_models / 'Unet_resnet34_valid_loss_0.325_epoch3.pth'
    model1 = get_model(model_type='Unet', encoder_name='resnet34', model_file=model_file1)
    model_dict1 = dict(model=model1, model_weight=0.8, image_shape=image_shape, activation='sigmoid', batch_size=64)
    model_dicts.append(model_dict1)

    model_file1 = path_models / 'UnetPlusPlus_densenet121_valid_loss_0.321_epoch2.pth'
    model1 = get_model(model_type='UnetPlusPlus', encoder_name='densenet121', model_file=model_file1)
    model_dict1 = dict(model=model1, model_weight=0.8, image_shape=image_shape, activation='sigmoid', batch_size=32)
    model_dicts.append(model_dict1)

    model_file1 = path_models / 'UnetPlusPlus_resnet34_valid_loss_0.331_epoch3.pth'
    model1 = get_model(model_type='UnetPlusPlus', encoder_name='resnet34', model_file=model_file1)
    model_dict1 = dict(model=model1, model_weight=0.7, image_shape=image_shape, activation='sigmoid', batch_size=32)
    model_dicts.append(model_dict1)

if args.task_type == 'ACDC2019_v2':
    path_models = Path(__file__).resolve().parent / 'trained_models' / 'v2' / 'dice'

    model_file1 = path_models / 'Unet_densenet121_valid_loss_0.328_epoch4.pth'
    model1 = get_model(model_type='Unet', encoder_name='densenet121', model_file=model_file1)
    model_dict1 = dict(model=model1, model_weight=1, image_shape=image_shape, activation='sigmoid', batch_size=32)
    model_dicts.append(model_dict1)

    model_file1 = path_models / 'Unet_resnet34_valid_loss_0.338_epoch4.pth'
    model1 = get_model(model_type='Unet', encoder_name='resnet34', model_file=model_file1)
    model_dict1 = dict(model=model1, model_weight=0.8, image_shape=image_shape, activation='sigmoid', batch_size=64)
    model_dicts.append(model_dict1)

    model_file1 = path_models / 'UnetPlusPlus_densenet121_valid_loss_0.33_epoch4.pth'
    model1 = get_model(model_type='UnetPlusPlus', encoder_name='densenet121', model_file=model_file1)
    model_dict1 = dict(model=model1, model_weight=0.8, image_shape=image_shape, activation='sigmoid', batch_size=32)
    model_dicts.append(model_dict1)

    model_file1 = path_models / 'UnetPlusPlus_resnet34_valid_loss_0.35_epoch5.pth'
    model1 = get_model(model_type='UnetPlusPlus', encoder_name='resnet34', model_file=model_file1)
    model_dict1 = dict(model=model1, model_weight=0.7, image_shape=image_shape, activation='sigmoid', batch_size=32)
    model_dicts.append(model_dict1)


#endregion

mask_threshold = 127

#region predict
path_csv = Path(__file__).resolve().parent / 'datafiles'
csv_valid = path_csv / f'{args.csv_file}'

def get_test_wsi():  #get wsi images only in the test dataset.
    list_wsi = []

    path_csv = Path(__file__).resolve().parent / 'datafiles'
    path_csv.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_valid)
    for _, row in df.iterrows():
        img_file = row['images']
        wsi_name = img_file.split('/')[-2]
        if wsi_name not in list_wsi:
            list_wsi.append(wsi_name)

    return list_wsi

list_wsi = get_test_wsi()

total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
for wsi in list_wsi:

    path_save = Path(args.path_output)
    path_save.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(str(csv_valid))
    df = df.query(f'images.str.contains("{wsi}") and not images.str.contains("random_")', engine='python')
    csv_tmp = path_save / 'temp.csv'
    df.to_csv(csv_tmp, index=False)

    list_data_loaders = []
    for model_dict in model_dicts:
        ds_valid = Dataset_CSV_SEM_SEG(csv_file=str(csv_tmp), image_shape=model_dict['image_shape'],
                                       mask_threshold=mask_threshold)
        dataloader_valid = DataLoader(ds_valid, batch_size=model_dict1['batch_size'], pin_memory=True)
        list_data_loaders.append(dataloader_valid)

    _, pred_masks = predict_multi_models(list_data_loaders, model_dicts, model_convert_gpu=True)

    list_masks = []
    for batch_idx, (images, masks) in enumerate(dataloader_valid):
        masks = masks.detach().cpu().numpy()
        list_masks.append(masks)
    masks = np.vstack(list_masks)
    gt_masks = np.vstack(list_masks)

    tp, tn, fp, fn = get_confusion_matrix(pred_masks, gt_masks, threshold=0.5)
    total_tp += tp
    total_tn += tn
    total_fp += fp
    total_fn += fn

#endregion



acc_valid, sen_valid, spe_valid, iou_valid, dice_valid =\
    get_metrics(total_tp, total_tn, total_fp, total_fn)
print(f'acc:{acc_valid:5.3f}, sen:{sen_valid:5.3f}, spe:{spe_valid:5.3f}')
print(f'iou:{iou_valid:5.3f}, dice:{dice_valid:5.3f}')

for index in range(pred_masks.shape[0]):
    img_file = df.values[index][0]
    mask_file = df.values[index][1]
    image = cv2.imread(img_file)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    pred_mask = pred_masks[index, 0, :, :] * 255

    cv2.imwrite(str(path_save / (Path(mask_file).stem+'pred.jpg')), pred_mask )
    cv2.imwrite(str(path_save / Path(img_file).name), image)
    cv2.imwrite(str(path_save / Path(mask_file).name), mask)


print('OK')