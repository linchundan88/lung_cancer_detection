'''
  generating csv files, which act as a parameter to the class Dataset_CSV_sem_seg constructor, based on patches files.
  data split into train validation and test dataset based on a patient level split.
'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from libs.dataPreprocess.my_data import get_patiend_id, split_patient_ids, write_csv
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_type', default='ACDC2019_v2')
parser.add_argument('--path_patches', default='/disk_data/data/ACDC_2019/Training/patches/level1_512')
#patches level1_512 should be level1_51, but I do not rename dirs.
args = parser.parse_args()

path_patches = Path(args.path_patches)

list_patient_id = get_patiend_id(path_patches)
list_patient_id_train, list_patient_id_valid, list_patient_id_test\
    = split_patient_ids(list_patient_id, valid_ratio=0.1, test_ratio=0.1, random_seed=1234)


path_csv = Path(__file__).resolve().parent.parent / 'datafiles'
path_csv.mkdir(parents=True, exist_ok=True)
csv_train = path_csv / f'{args.data_type}_train.csv'
csv_valid = path_csv / f'{args.data_type}_valid.csv'
csv_test = path_csv / f'{args.data_type}_test.csv'


write_csv(csv_train, path_patches, list_patient_id_train)
write_csv(csv_valid, path_patches, list_patient_id_valid)
write_csv(csv_test, path_patches, list_patient_id_test)


print('OK')

