'''
predict_one_patch:
predict_csv: used by predict_one_wsi.  create a data_loader based on the csv file.
predict_one_wsi: The following processes were executed to predict a WSI image: generating patches generating csv file,
   predict csv file, write predicted mask files(using predict_csv), add missing patches and combining patches to WSI file
'''
import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from libs.neuralNetworks.my_dataset import Dataset_CSV_SEM_SEG
from libs.images.my_img_to_tensor import img_to_tensor
from libs.wsi.my_patches import gen_patches, add_missing_patches, combine_patches
from libs.dataPreprocess.my_data import write_csv
import openslide
from math import ceil
from tqdm import tqdm
import time
import logging


@torch.no_grad()
def predict_csv(filename_csv, model_dicts, model_convert_gpu=True, num_workers=8, include_input=False):
    list_outputs = []

    for model_dict in model_dicts:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model_dict['model']
        if model_convert_gpu and torch.cuda.device_count() > 0:
            model.to(device)
        if model_convert_gpu and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.eval()

        dataset = Dataset_CSV_SEM_SEG(csv_file=filename_csv, image_shape=model_dict['image_shape'], test_mode=True)
        data_loader = DataLoader(dataset, batch_size=model_dict['batch_size'], num_workers=num_workers)

        list_batch_outputs = []
        list_batch_inputs = []
        for batch_idx, inputs in enumerate(tqdm(data_loader)):
            # print('batch:', batch_idx)
            inputs = inputs.to(device)
            outputs = model(inputs)
            if model_dict['activation'] == 'sigmoid':
                outputs = torch.sigmoid(outputs)

            if include_input:
                list_batch_inputs.append(inputs.cpu().numpy())
            list_batch_outputs.append(outputs.cpu().numpy())

        outputs = np.vstack(list_batch_outputs)  # Equivalent to np.concatenate(list_outputs, axis=0)
        list_outputs.append(outputs)

        # del model
        if torch.cuda.device_count() > 0:
            torch.cuda.empty_cache()

    for index, (model_dict, outputs) in enumerate(zip(model_dicts, list_outputs)):
        if index == 0:  # if 'probs_total' not in locals().keys():
            ensemble_outputs = outputs * model_dict['model_weight']
            total_weights = model_dict['model_weight']
        else:
            ensemble_outputs += outputs * model_dict['model_weight']
            total_weights += model_dict['model_weight']

    ensemble_outputs /= total_weights

    if not include_input:
        return list_outputs, ensemble_outputs
    else:
        return list_outputs, ensemble_outputs, np.vstack(list_batch_inputs)



def predict_one_wsi(path_wsi, model_dicts, slide_level, patch_h, patch_w, image_thresholds, path_output,
                    p_add_missing_patches=True, p_combine_patches=True, mask_threshold=127, num_workers=8):
    #generating patches
    path_patches = path_output / 'patches'
    path_patches.mkdir(parents=True, exist_ok=True)
    gen_patches(path_wsi=str(path_wsi), slide_level=slide_level, patch_h=patch_h, patch_w=patch_w,
                image_thresholds=image_thresholds, patches_dir=path_patches, gen_grid_patches=True)

    #generating csv file
    path_csv = path_output / f'predict.csv'
    write_csv(path_csv, path_patches, mask_should_exist=False)

    #predict patches
    logging.debug(f'start predicting models at time:{time.time()}.')
    _, outputs = predict_csv(str(path_csv), model_dicts, num_workers=num_workers)
    logging.debug(f'complete predicting models at time:{time.time()}.')

    # write predicted mask files
    df = pd.read_csv(str(path_csv))
    for index, row in df.iterrows():
        img_pred_mask = outputs[index, 0, :, :]  # (N,C,H,W)
        img_pred_mask *= 255

        img_file_mask = path_patches / (Path(row['images']).stem + '_pred_mask.jpg')
        print(img_file_mask)
        cv2.imwrite(str(img_file_mask), img_pred_mask)

        _, img_thres = cv2.threshold(img_pred_mask, mask_threshold, 255, cv2.THRESH_BINARY)  #mask_threshold 127
        img_file_binary_mask = path_patches / (Path(row['images']).stem + '_pred_binary_mask.jpg')
        print(img_file_binary_mask)
        cv2.imwrite(str(img_file_binary_mask), img_thres)

    if p_add_missing_patches:
        slide_wsi = openslide.OpenSlide(str(path_wsi))
        img_w, img_h = slide_wsi.dimensions
        downsampling = int(slide_wsi.level_downsamples[slide_level])  # if slide_level==2, downsampling=4
        patch_h_level0, patch_w_level0 = patch_h * int(downsampling), patch_w * int(downsampling)
        num_h, num_w = ceil(img_h / patch_h_level0), ceil(img_w / patch_w_level0)
        slide_wsi.close()

        img_black = np.zeros((patch_h, patch_w, 3))  # do net need to reconstruct WSI.
        add_missing_patches(path_patches, num_h, num_w, img_black, patch_tag='', file_ext='.jpg')
        img_black_mask = np.zeros((patch_h, patch_w))
        add_missing_patches(path_patches, num_h, num_w, img_black_mask, patch_tag='_pred_mask', file_ext='.jpg')
        add_missing_patches(path_patches, num_h, num_w, img_black_mask, patch_tag='_pred_binary_mask', file_ext='.jpg')

    if p_combine_patches:
        tiff_img = path_output / 'image.tif'
        combine_patches(path_patches, file_tiff=str(tiff_img), compress_ratio=95, patch_tag='', file_ext='.jpg')
        tiff_mask = path_output / 'pred_mask.tif'
        combine_patches(path_patches, file_tiff=str(tiff_mask), compress_ratio=95, patch_tag='_pred_mask', file_ext='.jpg')
        tiff_binary_mask = path_output / 'pred_binary_mask.tif'
        combine_patches(path_patches, file_tiff=str(tiff_binary_mask), compress_ratio=95, patch_tag='_pred_binary_mask', file_ext='.jpg')



@torch.no_grad()
def predict_one_patch(img_file, model_dicts, save_file=None, model_convert_gpu=True):
    assert os.path.exists(img_file), 'fine not found!'

    list_outputs = []

    for index, model_dict in enumerate(model_dicts):
        model = model_dict['model']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_convert_gpu and torch.cuda.device_count() > 0:
            model.to(device)  # model.cuda()
        model.eval()

        if index == 0:  # Reduce the number of reading image files
            image_shape = model_dict['image_shape']
            inputs = img_to_tensor(img_file, image_shape=model_dict['image_shape'])
            inputs = inputs.to(device)
        else:
            if model_dict['image_shape'] != image_shape:
                inputs = img_to_tensor(img_file, image_shape=model_dict['image_shape'])
                inputs = inputs.to(device)

        outputs = model(inputs)
        if model_dict['activation'] == 'sigmoid':
            outputs = torch.sigmoid(outputs)

        outputs = outputs.cpu().numpy()
        list_outputs.append(outputs)

    for index, (model_dict, outputs) in enumerate(zip(model_dicts, list_outputs)):
        if index == 0:  # if 'probs_total' not in locals().keys():
            ensemble_outputs = outputs * model_dict['model_weight']
            total_weights = model_dict['model_weight']
        else:
            ensemble_outputs += outputs * model_dict['model_weight']
            total_weights += model_dict['model_weight']

    ensemble_outputs /= total_weights


    if save_file:
        img_pred_mask = ensemble_outputs[0, 0, :, :]  # (N,C,H,W)
        _, img_thres = cv2.threshold(img_pred_mask, 0.5, 255, cv2.THRESH_BINARY)
        cv2.imwrite(save_file, img_thres)

    return list_outputs, ensemble_outputs



# if __name__ == '__main__':
#     print('OK')