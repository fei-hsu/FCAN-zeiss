from .resnet_encoder import *
from .skip_decoder import *
from .unet_parts import *
from .unet import *
from .pyramid_pooling import *
from .aspp_layer import *
from .deep_lab import *
from .msc import *

def DeepLab_ResNet101_MSC(n_classes):
    return MSC(
        base=DeepLabV2(
            n_classes=n_classes, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24]
        ),
        scales=[0.5, 0.75],
    )

