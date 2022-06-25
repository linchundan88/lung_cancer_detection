import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd


path_csv = Path(__file__).resolve().parent.parent / 'datafiles'
path_csv.mkdir(parents=True, exist_ok=True)
csv_test = path_csv / f'ACDC2019_v2_test.csv'

list_wsi = []

df = pd.read_csv(csv_test)
for _, row in df.iterrows():
    img_file = row['images']
    wsi_name = img_file.split('/')[-2]
    if wsi_name not in list_wsi:
        list_wsi.append(wsi_name)

print(list_wsi)


print('OK')