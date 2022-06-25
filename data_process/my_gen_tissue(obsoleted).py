'''
    This file is obsoleted. For training data, it is replaced by roi masks.
    During validation, image_thresholds are used to filter black images.
'''
from libs.wsi.gen_mask import gen_tissue_mask_whole
import os
from multiprocessing import Pool


wsi_dir = '/devdata_b/data/ACDC_2019/Training/images'
tissue_mask_dir = '/devdata_b/data/ACDC_2019/Training/tissue_masks/'

level = 0
multi_processes_num = 8
pool = Pool(processes=multi_processes_num)

for dir_path, subpaths, files in os.walk(wsi_dir, False):
    for f in files:
        file_wsi = os.path.join(dir_path, f)
        _, filename = os.path.split(file_wsi)
        filename_base, file_ext = os.path.splitext(filename)
        if file_ext.upper() not in ['.TIF']:
            continue

        # file_mask = os.path.join(tissue_dir, filename_base + f'_tissue_masklevel{level}.tif')
        file_mask = os.path.join(tissue_mask_dir, filename_base + f'_tissue_masklevel.tif')
        # gen_tissue_mask(file_wsi, level, file_mask)
        # file_mask = '/devdata_b/data/ACDC_2019/a0.tif'
        # gen_tissue_mask_whole(file_wsi, level, file_mask)
        pool.apply_async(gen_tissue_mask_whole, args=(file_wsi, level, file_mask))

pool.close()
pool.join()

print('OK')

