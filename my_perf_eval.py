'''
  Compute performance metrics on the dataset
  ACC, Sen, Spe, F1, IOU, Dice
'''

import cv2
from pathlib import Path
from libs.neuralNetworks.semanticSegmentation.metrics.metrics_numpy import get_confusion_matrix, get_metrics, get_iou, get_dice
import openslide
from math import ceil

path_wsi = Path('/disk_data/data/ACDC_2019/Training/predict_results/')
path_patches_base = ''

LIST_THRESHOLDS = [i/100 for i in range(5, 95, step=5)]

(patch_h, patch_w) = (512, 512)
slide_level = 2

for threshold in LIST_THRESHOLDS:
    (TP_num, TN_num, FP_num, FN_num) = (0, 0, 0, 0)

    for file_wsi in Path.rglob('*.tif'):
        slide_wsi = openslide.OpenSlide(str(file_wsi))
        img_w, img_h = slide_wsi.dimensions
        num_h = ceil(img_h / patch_h)
        num_w = (img_w / patch_w)

        path_patches = Path('patch_dir')
        for file_mask in path_patches.rglob('*_mask.jpg'):

            file_pred_mask = path_pred_mask / file_mask.parts[2].split('_')[0] / file_mask.parts[2].split('_')[1] / file_mask.stem + '_pred_mask' + file_mask.suffix
            if not file_pred_mask.exists():
                continue

            img_mask = cv2.imread(file_mask, cv2.IMREAD_GRAYSCALE)
            img_pred_mask = cv2.imread(file_pred_mask, cv2.IMREAD_GRAYSCALE)

            #h11_w23_mask.img  border patches ,crop  #crop_border  height weight
            tmp_file = file_mask.stem.replace('_mask', "")
            if num_h == tmp_file.split('_')[0].replace('h', ''):
                img_mask = img_mask[:]
            if num_w == tmp_file.split('_')[1].replace('2', ''):
                img_mask = img_mask[:, :]

            _, img1_mask = cv2.threshold(img_mask, 127, 1, cv2.THRESH_BINARY)
            _, img2_mask = cv2.threshold(img_pred_mask, 127, 1, cv2.THRESH_BINARY)

            #0， 1
            TP, TN, FP, FN = get_confusion_matrix(img1_mask, img2_mask, threshold=threshold)
            TP_num += TP
            TN_num += TN
            FP_num += FP
            FN_num += FN

    ACC, TPR, TNR, IOU, DICE = get_metrics(TP_num, TN_num, FP_num, FN_num)

    print(f'threshold:{threshold}')
    print(f'ACC:{ACC}, TPR:{TPR}, TNR:{TNR}, IOU:{IOU}, DICE:{DICE}')

print('OK')
