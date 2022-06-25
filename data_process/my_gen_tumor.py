'''
  Generating tumor mask whole slide images based on whole slide images and ASAP annotation xml files
'''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from libs.wsi.gen_mask import gen_mask
from multiprocessing import Pool
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_base', default='/disk_data/data/ACDC_2019/Training')
parser.add_argument('--multi_processes_num', type=int, default=8)
args = parser.parse_args()

path_base = Path(args.path_base)
path_wsi_base = path_base / 'images'
path_dest_base = path_base / 'tumor_masks'

pool = Pool(processes=args.multi_processes_num)

for path_wsi in path_wsi_base.rglob('*.tif'):
    path_xml = path_base / 'annotation' / path_wsi.parts[-2] / f'{path_wsi.stem}.xml'
    assert path_xml.exists(), f'file {path_xml} does not exists!'
    path_mask = path_dest_base / path_wsi.parts[-2] / f'{path_wsi.stem}_tumor_mask.tif'
    # gen_mask(str(path_wsi), str(path_xml), str(path_mask))
    pool.apply_async(gen_mask, args=(str(path_wsi), str(path_xml), str(path_mask)))

pool.close()
pool.join()

print('OK')

