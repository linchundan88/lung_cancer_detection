import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from multiprocessing import Pool
from libs.wsi.my_patches import gen_patches
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--slide_level', type=int, default=1)  #slide_level 1 2
parser.add_argument('--patch_h', type=int, default=512)  #patch_size (512, 512)
parser.add_argument('--patch_w', type=int, default=512)  #patch_size (512, 512)
parser.add_argument('--random_patch_ratio', type=float, default=0.3)
parser.add_argument('--path_base', default='/disk_data/data/ACDC_2019/Training')
parser.add_argument('--multi_processes_num', type=int, default=8)
args = parser.parse_args()


path_base = Path(args.path_base)
path_wsi_base = path_base / 'images'
path_tumor_mask_base = path_base / 'tumor_masks'
path_roi_mask_base = path_base / 'roi_masks'
path_patches_base = path_base / 'patches' / f'level{args.slide_level}_{args.patch_w}'

gen_grid_patches = True

image_thresholds = (1, 254.)
roi_thresholds = (2 / 255, 1)  # 0-1

ignore_border_patches=False
adding_border_padding=True

pool = Pool(processes=args.multi_processes_num)


for path_wsi in path_wsi_base.rglob('*.tif'):
    path_roi_mask = path_roi_mask_base / path_wsi.parts[-2] / f'{path_wsi.stem}_roi_mask.tif'
    assert path_roi_mask.exists(), f'file  {path_roi_mask} does not exists!'
    path_tumor_mask = path_tumor_mask_base / path_wsi.parts[-2] / f'{path_wsi.stem}_tumor_mask.tif'
    assert path_tumor_mask.exists(), f'file {path_tumor_mask} does not exists!'
    path_patches = path_patches_base / f'{path_wsi.parts[-2]}_{path_wsi.stem}'  #merge training1 and training2

    pool.apply_async(gen_patches, args=(path_wsi, args.slide_level, args.patch_h, args.patch_w, path_patches,
                                        ignore_border_patches, adding_border_padding,
                                        path_roi_mask, roi_thresholds, gen_grid_patches, args.random_patch_ratio,
                                        image_thresholds, path_tumor_mask))
    # gen_patches(path_wsi, args.slide_level, args.patch_h, args.patch_w, path_patches,
    #             ignore_border_patches, adding_border_padding,
    #             path_roi_mask, roi_thresholds, gen_grid_patches, args.random_patch_ratio,
    #             image_thresholds, path_tumor_mask)



pool.close()
pool.join()

print('OK')