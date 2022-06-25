import torch
from segmentation_models_pytorch import Unet, UnetPlusPlus
from libs.neuralNetworks.semanticSegmentation.models.u_net_variants import R2U_Net, AttU_Net, R2AttU_Net
from libs.neuralNetworks.semanticSegmentation.models.u2_net import U2NET, U2NETP


def get_model(model_type, encoder_name=None, model_file=None):

    dict_param = dict(encoder_weights='imagenet', encoder_name=encoder_name, classes=1, in_channels=3, activation=None)
    if model_type == 'Unet':
        model = Unet(encoder_depth=5, **dict_param)
    if model_type == 'UnetPlusPlus':
        model = UnetPlusPlus(encoder_depth=5, **dict_param)


    if model_type == 'R2U_Net':
        model = R2U_Net(img_ch=3, output_ch=1)
    if model_type == 'AttU_Net':
        model = AttU_Net(img_ch=3, output_ch=1)
    if model_type == 'R2AttU_Net':
        model = R2AttU_Net(img_ch=3, output_ch=1)


    if model_type == 'U2_NET':
        model = U2NET(in_ch=3, out_ch=1)
    if model_type == 'U2NETP':
        model = U2NETP(in_ch=3, out_ch=1)

    if model_file is not None:
        state_dict = torch.load(model_file, map_location='cpu')
        model.load_state_dict(state_dict)

    return model


    # model = Unet('vgg16', encoder_weights='imagenet', encoder_depth=4, decoder_channels=[512, 256, 128, 64],  activation=None)
    # model = smp.Unet('resnet18', encoder_weights='imagenet', in_channels=3, encoder_depth=4,
    #                  decoder_channels=[128, 64, 32, 16], activation=None)
