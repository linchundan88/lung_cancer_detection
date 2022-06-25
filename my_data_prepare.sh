#!/bin/bash

path_base="/disk_data/data/ACDC_2019/Training"
multi_processes_num=10
#python ./data_process/my_gen_tumor.py  --path_base ${path_base} --multi_processes_num ${multi_processes_num}
#python ./data_process/my_gen_roi.py  --path_base ${path_base} --multi_processes_num ${multi_processes_num}
python ./data_process/my_gen_patches.py  --slide_level 2  --patch_h 512 --patch_w 512 --path_base ${path_base} --multi_processes_num ${multi_processes_num}
#python ./data_process/my_gen_csv.py  --path_patches ${path_base}/patches/level2_512 --data_type ACDC2019_v1


