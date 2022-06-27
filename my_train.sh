#!/bin/bash

TASK_TYPE="ACDC2019_v2"
IMAGE_SHAPE="512 512"
SAVE_MODEL_DIR="/disk_code/code/lung_cancer_detection/trained_models_2022_6_27"
TRAIN_TIMES=1


for ((i=0;i<TRAIN_TIMES;i++))
do

  for loss_function in  "dice" # "dice" "bce" "softbce"
  do
    for model_type in  "UnetPlusPlus" #"Unet"
    #https://github.com/qubvel/segmentation_models.pytorch#architectures
    #"densenet121" "resnet34" "resnext50_32x4d" "timm-regnetx_008" "timm-resnest26d" "mobilenet_v2" "resnet34"  "timm-mobilenetv3_small_100" "timm-mobilenetv3_large_075"
    do
      BATCH_SIZE=64
      if ((encoder_name == "densenet121"))
      then
        BATCH_SIZE=32
      fi

      for encoder_name in  "densenet121" #"resnet34"
      do
        echo "training ${TASK_TYPE} times:${i} model_type:${model_type} encoder_name:${encoder_name} loss:${loss_function}"
        python ./my_train.py --task_type ${TASK_TYPE} \
          --model_type ${model_type}  --encoder_name ${encoder_name} --image_shape ${IMAGE_SHAPE} --loss_function ${loss_function}  \
          --epochs_num 6 --batch_size $BATCH_SIZE  --amp  \
          --save_model_dir ${SAVE_MODEL_DIR}/${TASK_TYPE}/${model_type}_${encoder_name}_${loss_function}_times${i}
      done
    done



#    for model_type in "AttU_Net"   # "R2U_Net" "R2AttU_Net"  are extremely memory hungry.
#    do
#      echo "training ${TASK_TYPE} times:${i} model:${model_type} loss:${loss_function}"
#      python ./my_train.py --task_type ${TASK_TYPE}  \
#        --model_type ${model_type}  --image_shape ${IMAGE_SHAPE} --loss_function ${loss_function}  \
#        --epochs_num 6 --batch_size 32 --amp  \
#        --save_model_dir ${SAVE_MODEL_DIR}/${TASK_TYPE}/${model_type}_${loss_function}_times${i}
#    done

#    for model_type in "U2_NET"
#    do
#      echo "training ${TASK_TYPE} times:${i} model:${model_type} loss:${loss_function}"
#      python ./my_train.py --task_type ${TASK_TYPE}  \
#        --model_type ${model_type}  --image_shape ${IMAGE_SHAPE} --loss_function ${loss_function}  \
#        --epochs_num 6 --batch_size 32 --amp  \
#        --save_model_dir ${SAVE_MODEL_DIR}/${TASK_TYPE}/${model_type}_${loss_function}_times${i}
#    done

  done

done

