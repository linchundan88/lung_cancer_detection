from libs.wsi.my_patches import add_missing_patches
from pathlib import Path
import openslide
import numpy as np
from math import ceil

dir_patches = Path('/disk_data/data/ACDC_2019/Training/patches/test_Training1_2')
patch_h, patch_w= 512, 512
image_pixel_value = 0  # 0, 1, 255
img_black = np.ones((patch_h, patch_w, 3), dtype=int) * image_pixel_value

#region get num_h, num_w
path_wsi = Path('/disk_data/data/ACDC_2019/Training/images/Training1/2.tif')
slide_level = 2
slide_wsi = openslide.OpenSlide(str(path_wsi))
img_w, img_h = slide_wsi.dimensions
downsampling = int(slide_wsi.level_downsamples[slide_level])
patch_h_level0, patch_w_level0 = patch_h * int(downsampling), patch_w * int(downsampling)

num_h = ceil(img_h / patch_h_level0)
num_w = ceil(img_w / patch_w_level0)

#endregion

add_missing_patches(dir_patches, num_h=num_h, num_w=num_w, img_black=img_black, patch_tag='', file_ext='jpg')



print('OK')
