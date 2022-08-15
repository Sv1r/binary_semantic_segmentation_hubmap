import segmentation_models_pytorch
print(f'SMP version: {segmentation_models_pytorch.__version__}\n')

import settings

manet = segmentation_models_pytorch.MAnet(
    encoder_name='mobilenet_v2',
    encoder_weights='imagenet',
    in_channels=settings.NUMBER_OF_INPUT_CHANNELS,
    classes=settings.NUMBER_OF_CLASSES,
    activation='sigmoid'
)
deeplab_p = segmentation_models_pytorch.DeepLabV3Plus(
    encoder_name='mobilenet_v2',
    encoder_weights='imagenet',
    in_channels=settings.NUMBER_OF_INPUT_CHANNELS,
    classes=settings.NUMBER_OF_CLASSES,
    activation='sigmoid'
)
unet_pp = segmentation_models_pytorch.UnetPlusPlus(
    encoder_name='mobilenet_v2',
    encoder_weights='imagenet',
    in_channels=settings.NUMBER_OF_INPUT_CHANNELS,
    classes=settings.NUMBER_OF_CLASSES,
    activation='sigmoid'
)
models_list = [manet, deeplab_p, unet_pp]
models_name_list = ['MaNet', 'DeepLab_P', 'Unet_PP']
model_dict = dict(zip(models_name_list, models_list))
