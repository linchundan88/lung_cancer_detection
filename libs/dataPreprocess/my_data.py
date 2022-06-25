'''
method get_patiend_id, split_patient_ids and write_csv are used for generating training csv files.
   split based on patient level
method write_csv_image_threshold is used for generating inference csv files:
'''
import random
import csv
import cv2
from libs.images.my_img import image_in_thresholds


def get_patiend_id(path1):
    list_pat_ids = []
    for item in path1.glob('**'):
        if item == path1:
            continue
        list_pat_ids.append(item.parts[-1])
    return list_pat_ids


def split_patient_ids(list_patient_id, valid_ratio=0.1, test_ratio=0.1,
                      random_seed=18888):
    random.seed(random_seed)
    random.shuffle(list_patient_id)

    if test_ratio is None:
        split_num = int(len(list_patient_id) * (1 - valid_ratio))
        list_patient_id_train = list_patient_id[:split_num]
        list_patient_id_valid = list_patient_id[split_num:]

        return list_patient_id_train, list_patient_id_valid
    else:
        split_num_train = int(len(list_patient_id) * (1 - valid_ratio - test_ratio))
        list_patient_id_train = list_patient_id[:split_num_train]
        split_num_valid = int(len(list_patient_id) * (1 - test_ratio))
        list_patient_id_valid = list_patient_id[split_num_train:split_num_valid]
        list_patient_id_test = list_patient_id[split_num_valid:]

        return list_patient_id_train, list_patient_id_valid, list_patient_id_test



def write_csv(path_csv, path_patches, list_patient=None, mask_should_exist=True):
    with open(str(path_csv), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])

        for file_img in path_patches.rglob('*.jpg'):
            if file_img.stem.endswith('_mask'):
                continue

            if list_patient is not None:
                patient_id = file_img.parts[-2]  # get Training2_23 from .../Training2_23/h19_w15.jpg
                if patient_id not in list_patient:
                    continue

            file_mask = file_img.parent / f'{file_img.stem}_mask{file_img.suffix}'
            if mask_should_exist and not file_mask.exists():
                continue

            print(file_img, file_mask)
            csv_writer.writerow([str(file_img), str(file_mask)])



def write_csv_image_threshold(path_csv, path_patches, with_mask=True, image_thresholds=None):
    with open(str(path_csv), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'masks'])

        for file_img in path_patches.rglob('*.jpg'):
            if file_img.stem.endswith('_mask'):
                continue

            if image_thresholds is not None:
                img1 = cv2.imread(str(file_img))
                if not image_in_thresholds(img1, image_thresholds):
                    continue

            if with_mask:
                file_mask = file_img.parent / f'{file_img.stem}_mask{file_img.suffix}'
                if not file_mask.exists():
                    continue
            else:
                file_mask = ''

            print(file_img, file_mask)
            csv_writer.writerow([str(file_img), str(file_mask)])



''' obsoleted      
def split_dataset(filename_csv, valid_ratio=0.1, test_ratio=None,
                  shuffle=True, random_state=None, field_columns=['images', 'labels']):

    if filename_csv.endswith('.csv'):
        df = pd.read_csv(filename_csv)
    elif filename_csv.endswith('.xls') or filename_csv.endswith('.xlsx'):
        df = pd.read_excel(filename_csv)

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    if test_ratio is None:
        split_num = int(len(df)*(1-valid_ratio))
        data_train = df[:split_num]
        train_files = data_train[field_columns[0]].tolist()
        train_labels = data_train[field_columns[1]].tolist()

        data_valid = df[split_num:]
        valid_files = data_valid[field_columns[0]].tolist()
        valid_labels = data_valid[field_columns[1]].tolist()

        return train_files, train_labels, valid_files, valid_labels
    else:
        split_num_train = int(len(df) * (1 - valid_ratio - test_ratio))
        data_train = df[:split_num_train]
        train_files = data_train[field_columns[0]].tolist()
        train_labels = data_train[field_columns[1]].tolist()

        split_num_valid = int(len(df) * (1 - test_ratio))
        data_valid = df[split_num_train:split_num_valid]
        valid_files = data_valid[field_columns[0]].tolist()
        valid_labels = data_valid[field_columns[1]].tolist()

        data_test = df[split_num_valid:]
        test_files = data_test[field_columns[0]].tolist()
        test_labels = data_test[field_columns[1]].tolist()

        return train_files, train_labels, valid_files, valid_labels, test_files, test_labels

'''