'''The training file, this code can be invoked by my_train.sh through the command line.'''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--CUDA_VISIBLE_DEVICES', default='0,1')
parser.add_argument('--task_type', default='ACDC2019_v2')
parser.add_argument('--model_type', default='Unet')  #Unet UnetPlusPlus R2U_Net
parser.add_argument('--encoder_name', default='resnet34')  #mobilenet_v2 resnet34 timm-mobilenetv3_small_100 timm-mobilenetv3_large_075
parser.add_argument('--image_shape', nargs='+', type=int, default=(512, 512))  #patch_size (512, 512)
parser.add_argument('--mask_threshold', type=float, default=127)  #binary mask images
parser.add_argument('--loss_function', default='dice')   #dice, bce, softbce, combine_dice_bce
parser.add_argument('--pos_weight', type=float, default='4.')  # only used for bce loss
parser.add_argument('--smooth_factor', type=float, default='0.1')  # only used for softbce loss
parser.add_argument('--amp', action='store_true', default=True)  # AUTOMATIC MIXED PRECISION
#recommend num_workers = the number of gpus * 4, when debugging it should be set to 0.
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=64)  #DP:64, DDP:32
parser.add_argument('--weight_decay', type=float, default=0)  #L2 regularization, 1e-4 is too big
parser.add_argument('--epochs_num', type=int, default=4)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--save_model_dir', default='/tmp')
parser.add_argument('--parallel_mode', default='DP')   # DP:Data Parallel,  DDP:Distributed Data Data Parallel
args = parser.parse_args()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES   # setting  GPUs, must before import torch
import torch
from libs.neuralNetworks.semanticSegmentation.models.my_get_model_helper import get_model
from libs.neuralNetworks.my_dataset import Dataset_CSV_SEM_SEG
import torch.optim as optim
# import torch_optimizer as optim_plus
from torch.optim.lr_scheduler import StepLR
import albumentations as A
# from libs.neuralNetworks.semanticSegmentation.losses.my_loss_sem_seg import DiceLoss
from segmentation_models_pytorch.losses.dice import DiceLoss
from segmentation_models_pytorch.losses.soft_bce import SoftBCEWithLogitsLoss
from libs.neuralNetworks.semanticSegmentation.losses.my_ce_dice import CE_Dice_combine_Loss
from libs.neuralNetworks.semanticSegmentation.my_train_helper import my_train
from munch import Munch



#region prepare dataset
path_csv = Path(__file__).resolve().parent / 'datafiles'
csv_train = path_csv / f'{args.task_type}_train.csv'
csv_valid = path_csv / f'{args.task_type}_valid.csv'

transform_train = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    # A.RandomCrop(width=args.image_shape[1], height=args.image_shape[0]),
    A.RandomRotate90(p=0.8),
    # A.ShiftScaleRotate(p=0.8, rotate_limit=(-10, 10)),
    A.Resize(args.image_shape[0], args.image_shape[1]),  #(height,weight)  if random cropping patch size is different from ...
    # A.ShiftScaleRotate(p=0.1, shift_limit=0.05, scale_limit=0.1, rotate_limit=10),
    # A.Affine(scale=0.1, rotate=10, translate_percent=0.1),
    # A.RandomBrightnessContrast(p=0.8, brightness_limit=0.1, contrast_limit=0.1),
    # A.gaussian_blur(p=0.8, blur_limit=(2, 4), sigma=(0.1, 3))
])


ds_train = Dataset_CSV_SEM_SEG(csv_file=csv_train, transform=transform_train,
                               image_shape=args.image_shape, mask_threshold=args.mask_threshold) #image_shape=args.image_shape,
ds_valid = Dataset_CSV_SEM_SEG(csv_file=csv_valid, image_shape=args.image_shape, mask_threshold=args.mask_threshold)  #

#endregion


#region training

model = get_model(model_type=args.model_type, encoder_name=args.encoder_name)

if args.loss_function == 'dice':
    criterion = DiceLoss(mode='binary', from_logits=True)  # segmentation_models_pytorch loss function
if args.loss_function == 'bce':
    pos_weight = torch.FloatTensor(torch.tensor([args.pos_weight]))
    if torch.cuda.is_available():
        pos_weight = pos_weight.cuda()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
if args.loss_function == 'combine_dice_bce':
    pos_weight = torch.FloatTensor(torch.tensor([args.pos_weight]))
    if torch.cuda.is_available():
        pos_weight = pos_weight.cuda()
    criterion = CE_Dice_combine_Loss(weight_bce=1, weight_dice=1, pos_weight=pos_weight)
if args.loss_function == 'softbce':
    pos_weight = torch.FloatTensor(torch.tensor([args.pos_weight]))
    if torch.cuda.is_available():
        pos_weight = pos_weight.cuda()
    criterion = SoftBCEWithLogitsLoss(pos_weight=pos_weight, smooth_factor=args.smooth_factor)
# criterion = DiceLoss(activation='sigmoid')  #My loss function


optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# optimizer = optim_plus.radam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)
# optimizer = optim_plus.Lookahead(optimizer, k=5, alpha=0.5)
scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
# scheduler = CosineAnnealingLR(optimizer, T_max=epochs_num // 4, eta_min=0)  #T_max: half of one circle
# scheduler = GradualWarmupScheduler(optimizer, multiplier=4, total_epoch=2, after_scheduler=scheduler)


#Munch is better than Dict. The contents of it can be accessed by dot operator or string name.
train_config = Munch({
    'model': model, 'ds_train': ds_train, 'batch_size': args.batch_size,
    'num_workers': args.num_workers, 'criterion': criterion,
    'accumulate_grads_times': None,
    'optimizer': optimizer, 'scheduler': scheduler,
    'amp': args.amp, 'epochs_num': args.epochs_num,
    'list_ds_valid': [ds_valid],
    'activation': 'sigmoid',
    'save_model_dir': Path(args.save_model_dir),
})


if __name__ == "__main__":
    if args.parallel_mode == 'DP':
        my_train(train_config)
    if args.parallel_mode == 'DDP':
        import torch.multiprocessing as mp
        from libs.neuralNetworks.semanticSegmentation.my_train_ddp_helper import main_worker
        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
        mp.spawn(main_worker, args=(n_gpus, train_config), nprocs=n_gpus, join=True)

    print('OK')

#endregion




