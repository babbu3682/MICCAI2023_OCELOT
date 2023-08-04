import torch 
import torch.nn as nn
import torch.nn.functional as F

from arch.unet import UNet
from arch.our_network import *

def get_model(name):

    if name == "Cell_UNet":
        model = UNet()

    elif name == 'Cell_MaxViT_UNet_MTL_DET_SEG_REC':
        model = Cell_MaxViT_UNet_MTL_DET_SEG_REC()    

    elif name == 'Cell_MaxViT_UNet_MTL_DET_SEG_POI':
        model = Cell_MaxViT_UNet_MTL_DET_SEG_POI()

    elif name == 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI':
        model = Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI()   

    elif name == 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method':
        model = Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method()           

    elif name == 'Cell_MaxViT_UNet_DET':
        model = Cell_MaxViT_UNet_DET()                   

    elif name == 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Original':
        model = Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Original()       

    # elif name == 'Cell_MaxViT_UNet_MTL_Cross_All':
    #     model = Cell_MaxViT_UNet_MTL_Cross_All()   

    # elif name == 'Cell_MaxViT_UNet_MTL_Cross_Conv':
    #     model = Cell_MaxViT_UNet_MTL_Cross_Conv()   

    # elif name == 'Cell_MaxViT_UNet_MTL_Cross_Conv_Point':
    #     model = Cell_MaxViT_UNet_MTL_Cross_Conv_Point()

    # elif name == 'Cell_MaxViT_UNet_MTL_Cross_ResALL':
    #     model = Cell_MaxViT_UNet_MTL_Cross_ResALL() 

    # elif name == 'Cell_MaxViT_UNet_MTL_Cross_ResALL_V2':
    #     model = Cell_MaxViT_UNet_MTL_Cross_ResALL_V2()         

    # elif name == 'Cell_MaxViT_UNet_MTL_Cross_ResALL_V3':
    #     model = Cell_MaxViT_UNet_MTL_Cross_ResALL_V3()                 


    # print number of learnable parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of Learnable Params:', n_parameters)   

    return model


