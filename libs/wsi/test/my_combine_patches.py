from libs.wsi.my_patches import  combine_patches
from pathlib import Path
import openslide


dir_patches = Path('/disk_data/data/ACDC_2019/Training/patches/test_Training1_2')
file_tiff = Path('/disk_data/data/ACDC_2019/Training/Training1_2.tif')


combine_patches(dir_patches, scale_ratio=1, file_tiff=file_tiff, compress_ratio=95, patch_tag='')



slide_wsi = openslide.OpenSlide(str(file_tiff))
img_w, img_h = slide_wsi.dimensions

print(img_w, img_h)

print('OK')