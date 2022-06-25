#!/bin/bash

task_type="ACDC2019_v1"
slide_level="1"
path_wsi="/disk_data/data/lung_cancer_detection/Training/images"
path_output="/disk_data/data/lung_cancer_detection/Training/predict_results_2022_6_25/ACDC2019_v1"
python ./my_predict_holdout.py --path_wsi ${path_wsi} --path_output ${path_output}$ --task_type ${task_type} --slide_level ${slide_level}

task_type="ACDC2019_v2"
slide_level="2"
path_wsi="/disk_data/data/lung_cancer_detection/Training/images"
path_output="/disk_data/data/lung_cancer_detection/Training/predict_results_2022_6_25/ACDC2019_v2"
python ./my_predict_holdout.py --path_wsi ${path_wsi} --path_output ${path_output} --task_type ${task_type} --slide_level ${slide_level}


#task_type="ACDC2019_v1"
#slide_level="1"
#path_wsi="/disk_data/data/lung_cancer_detection/Testing/images"
#path_output="/disk_data/data/lung_cancer_detection/Training/predict_results_2022_6_18/ACDC2019_v1"
#python ./my_predict_test.py --path_wsi ${path_wsi} --path_output ${path_output}$ --task_type ${task_type} --slide_level ${slide_level}
#
#task_type="ACDC2019_v2"
#slide_level="2"
#path_wsi="/disk_data/data/lung_cancer_detection/Testing/images"
#path_output="/disk_data/data/lung_cancer_detection/Training/predict_results_2022_6_18/ACDC2019_v2"
#python ./my_predict_test.py --path_wsi ${path_wsi} --path_output ${path_output} --task_type ${task_type} --slide_level ${slide_level}