'''
gen_patches:  generating image and mask patches based on ROI(can be tissue and tumor areas) mask files.

add_missing_patches: add missing patches in a directory.
   During constructing the image folder, a lot of images that pixel values beyond pre-defined thresholds  were removed.

combine_patches： combining patches from one directory into a whole slide image tiff file.

'''

import gc
import os
import random
import cv2
import numpy as np
import openslide
from libs.images.my_img import save_tiled_tiff, image_in_thresholds
from math import ceil
from libs.images.my_img import image_pad
MAX_W=100000
MAX_H=100000


def _get_patch_bbox(img_h, img_w, patch_h, patch_w, gen_grid_patches=True, random_patch_ratio=0,
                    ignore_border_patches=False, adding_border_padding=True):
    list_bbox = list()

    if gen_grid_patches:
        for num_y in range(ceil(img_h / patch_h)):
            for num_x in range(ceil(img_w / patch_w)):
                y1 = num_y * patch_h
                y2 = y1 + patch_h
                x1 = num_x * patch_w
                x2 = x1 + patch_w

                if ignore_border_patches:
                    if y2 > img_h or x2 > img_w:
                        continue

                if not adding_border_padding:
                    if y2 > img_h:
                        y2 = img_h
                    if x2 > img_w:
                        x2 = img_w

                list_bbox.append((y1, y2, x1, x2, f'h{num_y}_w{num_x}'))

    if random_patch_ratio !=0:
        patch_random_num = int(len(list_bbox) * random_patch_ratio)
        for index in range(patch_random_num):
            y1 = random.randint(0, img_h - patch_h)
            y2 = y1 + patch_h
            x1 = random.randint(0, img_w - patch_w)
            x2 = x1 + patch_w

            if ignore_border_patches:
                if y2 > img_h or x2 > img_w:
                    continue

            if not adding_border_padding:
                if y2 > img_h:
                    y2 = img_h
                if x2 > img_w:
                    x2 = img_w

            list_bbox.append((y1, y2, x1, x2, f'random_{index}'))

    return list_bbox


def gen_patches(path_wsi, slide_level, patch_h, patch_w, patches_dir, ignore_border_patches=False, adding_border_padding=True,
                path_roi_wsi=None, roi_thresholds=None, gen_grid_patches=True, random_patch_ratio=0,
                image_thresholds=None,
                path_tumor_wsi=None):

    slide_wsi = openslide.OpenSlide(str(path_wsi))
    img_w, img_h = slide_wsi.dimensions
    downsampling = int(slide_wsi.level_downsamples[slide_level])  # if slide_level==2, downsampling=4
    patch_h_level0, patch_w_level0 = patch_h * int(downsampling), patch_w * int(downsampling)
    down_factor = 1 / downsampling
    patches_dir.mkdir(parents=True, exist_ok=True)

    if roi_thresholds is not None:
        slide_roi = openslide.OpenSlide(str(path_roi_wsi))
    if path_tumor_wsi is not None:
        slide_tumor = openslide.OpenSlide(str(path_tumor_wsi))

    list_bbox = _get_patch_bbox(img_h, img_w, patch_h_level0, patch_w_level0,
                                gen_grid_patches=gen_grid_patches, random_patch_ratio=random_patch_ratio,
                                ignore_border_patches=ignore_border_patches, adding_border_padding=adding_border_padding)

    for (y1, y2, x1, x2, file_stem) in list_bbox:
        if roi_thresholds is not None:
            # img_patch_roi = np.array(slide_roi.read_region((x1, y1), slide_level, (patch_w, patch_h)).convert('RGB'))
            # roi_patch = np.array(slide_roi.read_region((x1, y1), 0, (patch_w_level0, patch_h_level0)).convert('RGB'))
            roi_patch = np.array(slide_roi.read_region((x1, y1), 0, (x2-x1, y2-y1)).convert('RGB'))
            roi_patch = cv2.resize(roi_patch, None, fx=down_factor, fy=down_factor)
            if not image_in_thresholds(roi_patch, roi_thresholds):
                continue

        # img_patch = np.array(slide_wsi.read_region((x1, y1), slide_level, (patch_w, patch_h)).convert('RGB'))  #border
        img_patch = np.array(slide_wsi.read_region((x1, y1), slide_level, ((x2 - x1) // downsampling, (y2 - y1) // downsampling)).convert('RGB'))  # border
        # '/disk_data/data/ACDC_2019/Training/images/Training2/37.tif':  Training2/19 the following line raise jpeg error.
        # img_patch = np.array(slide_wsi.read_region((x1, y1), 0, (x2, y2)).convert('RGB'))
        # img_patch = cv2.resize(img_patch, None, fx=down_factor, fy=down_factor)
        if image_thresholds is not None:
            if not image_in_thresholds(img_patch, image_thresholds):
                continue

        file_patch_image = patches_dir / f'{file_stem}.jpg'
        img_patch = cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR)
        print(file_patch_image)
        cv2.imwrite(str(file_patch_image), img_patch)

        if path_tumor_wsi is not None:
            tumor_patch = np.array(slide_tumor.read_region((x1, y1), 0, (x2 - x1, y2 - y1)).convert('RGB'))
            tumor_patch = cv2.resize(tumor_patch, None, fx=down_factor, fy=down_factor)
            tumor_patch = cv2.cvtColor(tumor_patch * 255, cv2.COLOR_BGR2GRAY)  #gen_mask()  pixel value = 1

            file_patch_mask = patches_dir / f'{file_stem}_mask.jpg'
            print(file_patch_mask)
            cv2.imwrite(str(file_patch_mask), tumor_patch)

    slide_wsi.close()
    if path_tumor_wsi is not None:
        slide_roi.close()
    if path_tumor_wsi is not None:
        slide_tumor.close()
    gc.collect()


def add_missing_patches(dir_patches, num_h, num_w, img_black, patch_tag='', file_ext='.jpg'):
    for i in range(num_h):
        for j in range(num_w):
            file_patch = dir_patches / f'h{i}_w{j}{patch_tag}{file_ext}'  # image, mask
            if not file_patch.exists():  # h1_w9.jpg h1_w9_mask.jpg
                print(file_patch)
                cv2.imwrite(str(file_patch), img_black)



#if save tiff file, bgr_2_rgb should set to True.
def combine_patches(dir_patches, bgr_2_rgb=True, scale_ratio=1, gc_collect=True,
                    file_tiff=None, compress_ratio=None, patch_tag='', file_ext='.jpg'):

    list_patches_h = []
    for num_h in range(MAX_H):
        if not dir_patches.joinpath(f'h{num_h}_w0{file_ext}').exists():
            break

        list_patches_w = []
        for num_w in range(MAX_W):
            file_patch = dir_patches / f'h{num_h}_w{num_w}{patch_tag}{file_ext}'  #image, mask
            if not file_patch.exists():  #h1_w9.jpg
                break

            print(file_patch)
            img = cv2.imread(str(file_patch))
            if scale_ratio != 1:
                img = cv2.resize(img, None, fx=scale_ratio, fy=scale_ratio)
            if bgr_2_rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            list_patches_w.append(img)

        print('concatenating axis1')
        img_w = np.concatenate(list_patches_w, axis=1)
        if gc_collect:
            del list_patches_w
            gc.collect()
        list_patches_h.append(img_w)

    print('concatenating axis0')
    img1 = np.concatenate(list_patches_h, axis=0)
    if gc_collect:
        del list_patches_h
        gc.collect()


    if file_tiff is not None:
        print(f'save tiff file:{file_tiff}')
        save_tiled_tiff(str(file_tiff), img1, compress_ratio=compress_ratio)
        if gc_collect:
            del img1
            gc.collect()
    else:
        return img1






#test code
if __name__ == '__main__':
    from pathlib import Path

    roi_thresholds = (2 / 255, 1)
    image_thresholds = (0.8, 254.)
    slide_level = 2
    patch_h, patch_w = 512, 512
    path_wsi = Path('/disk_data/data/ACDC_2019/Training/images/Training1/3.tif')
    path_roi_mask = Path('/disk_data/data/ACDC_2019/Training/roi_masks/Training1/3_roi_mask.tif')
    path_patches = Path('/disk_data/data/ACDC_2019/Training/test1/Training1/3/patches')
    path_patches.mkdir(parents=True, exist_ok=True)

    gen_patches(path_wsi=path_wsi, slide_level=2, patch_h=512, patch_w=512,
                patches_dir=path_patches, path_roi_wsi=path_roi_mask, roi_thresholds=roi_thresholds, gen_grid_patches=True,
                random_patch_ratio=0, image_thresholds=image_thresholds, path_tumor_wsi=None)


    #add missing patches
    slide_wsi = openslide.OpenSlide(str(path_wsi))
    img_w, img_h = slide_wsi.dimensions
    downsampling = int(slide_wsi.level_downsamples[slide_level])  # if slide_level==2, downsampling=4
    patch_h_level0, patch_w_level0 = patch_h * int(downsampling), patch_w * int(downsampling)
    num_h, num_w = ceil(img_h / patch_h_level0), ceil(img_w / patch_w_level0)

    img_black = np.zeros((patch_h, patch_w, 3))  # do net need to reconstruct WSI.
    add_missing_patches(path_patches, num_h, num_w, img_black, patch_tag='', file_ext='.jpg')
    img_black_mask = np.zeros((patch_h, patch_w))
    add_missing_patches(path_patches, num_h, num_w, img_black_mask, patch_tag='', file_ext='.jpg')


    #combining patches to WSI file
    path_output = Path('/disk_data/data/ACDC_2019/Training/test1/Training1/3')
    file_tiff = path_output / 'image.tif'
    combine_patches(path_patches, file_tiff=str(file_tiff), compress_ratio=95, patch_tag='', file_ext='.jpg')



    print('OK')