import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1')
#ACDC2019_v2 slide_level=2, ACDC2019_v1 slide_level=1
parser.add_argument('--task_type', default='ACDC2019_v1')
parser.add_argument('--slide_level', default=1)
parser.add_argument('--path_wsi', default='/disk_data/data/ACDC_2019/Testing/images')
parser.add_argument('--path_output', default='/disk_data/data/ACDC_2019/Testing/predict_results_2022_6_27')
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
from libs.neuralNetworks.semanticSegmentation.models.my_get_model_helper import get_model
from libs.neuralNetworks.semanticSegmentation.my_predict_helper import predict_one_wsi
import time
import logging


#region get models
image_shape = (512, 512)

print('loading model..')
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


print('loading model completed.')
#endregion


patch_h, patch_w = (512, 512)
# image_thresholds = (0.4, 254.3)
image_thresholds = (1, 254.)

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='predict.log', level=logging.DEBUG, format=LOG_FORMAT)

for path_wsi in Path(args.path_wsi).rglob('*.tif'):
    logging.debug(f'start predicting {path_wsi}, time:{time.time()}')
    path_output = Path(args.path_output) / args.task_type / path_wsi.stem

    path_output.mkdir(parents=True, exist_ok=True)
    predict_one_wsi(path_wsi, model_dicts, args.slide_level, patch_h, patch_w, image_thresholds, path_output,
                    p_add_missing_patches=True, p_combine_patches=True)
    logging.debug(f'end predicting {path_wsi}, time:{time.time()}')



print('OK')
