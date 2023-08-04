import torch 
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Any, Iterator, Mapping
from itertools import chain

def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)

class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class Attention(nn.Module):
    def __init__(self, name, **params):
        super().__init__()
        if name is None:
            self.attention = nn.Identity(**params)
        elif name == "scse":
            self.attention = SCSEModule(**params)
        else:
            raise ValueError("Attention {} is not implemented".format(name))

    def forward(self, x):
        return self.attention(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


# upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
class DET_DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None, last=False):
        super().__init__()
        self.attention1 = nn.Identity() if last else Attention(attention_type, in_channels=in_channels+skip_channels)
        self.conv1      = Conv2dReLU(in_channels+skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2      = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class SEG_DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None, last=False):
        super().__init__()
        self.attention1 = nn.Identity() if last else Attention(attention_type, in_channels=in_channels+skip_channels)
        self.conv1      = Conv2dReLU(in_channels+skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2      = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, scale, input_channels, output_channels, ksize=1):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(input_channels, output_channels*(scale**2), kernel_size=1, stride=1, padding=ksize//2),
            nn.PixelShuffle(upscale_factor=scale)
        )

    def forward(self, input):
        return self.upsample(input)

class REC_DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
        super().__init__()
        self.upsample   = UpsampleBlock(scale=2, input_channels=in_channels, output_channels=in_channels)
        self.attention1 = Attention(attention_type, in_channels=in_channels)
        self.conv1      = Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2      = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x):
        x = self.upsample(x)
        x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class REC_Skip_DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, use_batchnorm=True, attention_type=None, last=False):
        super().__init__()
        self.upsample   = UpsampleBlock(scale=2, input_channels=in_channels, output_channels=in_channels)
        self.attention1 = nn.Identity() if last else Attention(attention_type, in_channels=in_channels+skip_channels)
        self.conv1      = Conv2dReLU(in_channels+skip_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.conv2      = Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.attention2 = Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

  

### MaxViT
class CrossStitch_Conv_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossStitch_Conv_Unit, self).__init__()
        self.a = Conv2dReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, use_batchnorm=True)
        self.b = Conv2dReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, use_batchnorm=True)
    
    def forward(self, x_a, x_b):
        mix = torch.concat([x_a, x_b], dim=1) # B, C, H, W
        new_x_a = self.a(mix)
        new_x_b = self.b(mix)
        return new_x_a, new_x_b

class CrossStitch_ConvRes_Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CrossStitch_ConvRes_Unit, self).__init__()
        self.a = Conv2dReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, use_batchnorm=True)
        self.b = Conv2dReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1, use_batchnorm=True)
    
    def forward(self, x_a, x_b):
        mix = torch.concat([x_a, x_b], dim=1)
        new_x_a = self.a(mix)
        new_x_b = self.b(mix)
        return new_x_a+x_a, new_x_b+x_b


class Cell_MaxViT_UNet_MTL_DET_SEG_REC(nn.Module):
    def __init__(self):
        super(Cell_MaxViT_UNet_MTL_DET_SEG_REC, self).__init__()

        # Encoder
        self.encoder = timm.create_model('maxvit_xlarge_tf_512.in21k_ft_in1k', pretrained=True, features_only=True) # Xlarge

        # DET Decoder
        self.det_decoder_block1 = DET_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block2 = DET_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block3 = DET_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block4 = DET_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block5 = DET_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse')

        # SEG Decoder
        self.seg_decoder_block1 = SEG_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block2 = SEG_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block3 = SEG_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block4 = SEG_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block5 = SEG_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse')
        
        # REC Decoder
        self.rec_decoder_block1 = REC_DecoderBlock(in_channels=1536,  out_channels=256, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block2 = REC_DecoderBlock(in_channels=256,   out_channels=128, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block3 = REC_DecoderBlock(in_channels=128,   out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block4 = REC_DecoderBlock(in_channels=64,    out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block5 = REC_DecoderBlock(in_channels=32,    out_channels=16,  use_batchnorm=True, attention_type='scse')

        # Head
        self.det_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.seg_head = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.rec_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.poi_head = nn.Linear(in_features=1536, out_features=2, bias=True)

        # Cross Stitch
        self.cross_stitch1 = CrossStitch_Conv_Unit(in_channels=256*2, out_channels=256)
        self.cross_stitch2 = CrossStitch_Conv_Unit(in_channels=128*2, out_channels=128)
        self.cross_stitch3 = CrossStitch_Conv_Unit(in_channels=64*2,  out_channels=64)
        self.cross_stitch4 = CrossStitch_Conv_Unit(in_channels=32*2,  out_channels=32)
        self.cross_stitch5 = CrossStitch_Conv_Unit(in_channels=16*2,  out_channels=16)

        # Init
        self.initialize()

    def initialize(self):
        initialize_decoder(self.det_decoder_block1)
        initialize_decoder(self.det_decoder_block2)
        initialize_decoder(self.det_decoder_block3)
        initialize_decoder(self.det_decoder_block4)
        initialize_decoder(self.det_decoder_block5)
        initialize_decoder(self.seg_decoder_block1)
        initialize_decoder(self.seg_decoder_block2)
        initialize_decoder(self.seg_decoder_block3)
        initialize_decoder(self.seg_decoder_block4)
        initialize_decoder(self.seg_decoder_block5)        
        initialize_decoder(self.rec_decoder_block1)
        initialize_decoder(self.rec_decoder_block2)
        initialize_decoder(self.rec_decoder_block3)
        initialize_decoder(self.rec_decoder_block4)
        initialize_decoder(self.rec_decoder_block5)  
        initialize_head(self.det_head)
        initialize_head(self.seg_head)
        initialize_head(self.rec_head)
        initialize_head(self.poi_head)


    def forward(self, x):
        # encoder
        skip4, skip3, skip2, skip1, x = self.encoder(x)
        
        det1 = self.det_decoder_block1(x,   skip1)
        seg1 = self.seg_decoder_block1(x,   skip1)
        det1, seg1 = self.cross_stitch1(det1, seg1)

        det2 = self.det_decoder_block2(det1, skip2)
        seg2 = self.seg_decoder_block2(seg1, skip2)
        det2, seg2 = self.cross_stitch2(det2, seg2)

        det3 = self.det_decoder_block3(det2, skip3)
        seg3 = self.seg_decoder_block3(seg2, skip3)
        det3, seg3 = self.cross_stitch3(det3, seg3)

        det4 = self.det_decoder_block4(det3, skip4)        
        seg4 = self.seg_decoder_block4(seg3, skip4)        
        det4, seg4 = self.cross_stitch4(det4, seg4)
        
        det5 = self.det_decoder_block5(det4)
        seg5 = self.seg_decoder_block5(seg4)
        det5, seg5 = self.cross_stitch5(det5, seg5)

        # rec decoder
        rec = self.rec_decoder_block1(x)
        rec = self.rec_decoder_block2(rec)
        rec = self.rec_decoder_block3(rec)
        rec = self.rec_decoder_block4(rec)        
        rec = self.rec_decoder_block5(rec)    

        # head
        det5 = self.det_head(det5)
        seg5 = self.seg_head(seg5)
        rec = self.rec_head(rec)

        return det5, seg5, rec

class Cell_MaxViT_UNet_MTL_DET_SEG_POI(nn.Module):
    def __init__(self):
        super(Cell_MaxViT_UNet_MTL_DET_SEG_POI, self).__init__()

        # Encoder
        self.encoder = timm.create_model('maxvit_xlarge_tf_512.in21k_ft_in1k', pretrained=True, features_only=True) # Xlarge

        # DET Decoder
        self.det_decoder_block1 = DET_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block2 = DET_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block3 = DET_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block4 = DET_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block5 = DET_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse')

        # SEG Decoder
        self.seg_decoder_block1 = SEG_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block2 = SEG_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block3 = SEG_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block4 = SEG_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block5 = SEG_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse')

        # POI Decoder   
        self.poi_decoder_block1 = nn.AdaptiveAvgPool2d(1)
        self.poi_decoder_block2 = LayerNorm2d(num_channels=1536, eps=1e-5, affine=True)
        self.poi_decoder_block3 = Flatten()
        self.poi_decoder_block4 = nn.Sequential(nn.Linear(in_features=1536, out_features=1536, bias=True), nn.Tanh())
        self.poi_decoder_block5 = nn.Dropout(p=0.2, inplace=False)

        # Head
        self.det_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.seg_head = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.rec_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.poi_head = nn.Linear(in_features=1536, out_features=2, bias=True)

        # Cross Stitch
        self.cross_stitch1 = CrossStitch_Conv_Unit(in_channels=256*2, out_channels=256)
        self.cross_stitch2 = CrossStitch_Conv_Unit(in_channels=128*2, out_channels=128)
        self.cross_stitch3 = CrossStitch_Conv_Unit(in_channels=64*2,  out_channels=64)
        self.cross_stitch4 = CrossStitch_Conv_Unit(in_channels=32*2,  out_channels=32)
        self.cross_stitch5 = CrossStitch_Conv_Unit(in_channels=16*2,  out_channels=16)

        # Init
        self.initialize()

    def initialize(self):
        initialize_decoder(self.det_decoder_block1)
        initialize_decoder(self.det_decoder_block2)
        initialize_decoder(self.det_decoder_block3)
        initialize_decoder(self.det_decoder_block4)
        initialize_decoder(self.det_decoder_block5)
        initialize_decoder(self.seg_decoder_block1)
        initialize_decoder(self.seg_decoder_block2)
        initialize_decoder(self.seg_decoder_block3)
        initialize_decoder(self.seg_decoder_block4)
        initialize_decoder(self.seg_decoder_block5)        
        initialize_head(self.det_head)
        initialize_head(self.seg_head)
        initialize_head(self.rec_head)
        initialize_head(self.poi_head)


    def forward(self, x):
        # encoder
        skip4, skip3, skip2, skip1, x = self.encoder(x)
        
        poi = self.poi_decoder_block1(x)
        poi = self.poi_decoder_block2(poi)
        poi = self.poi_decoder_block3(poi)
        poi = self.poi_decoder_block4(poi)
        poi = self.poi_decoder_block5(poi)

        det1 = self.det_decoder_block1(x,   skip1)
        seg1 = self.seg_decoder_block1(x,   skip1)
        det1, seg1 = self.cross_stitch1(det1, seg1)

        det2 = self.det_decoder_block2(det1, skip2)
        seg2 = self.seg_decoder_block2(seg1, skip2)
        det2, seg2 = self.cross_stitch2(det2, seg2)

        det3 = self.det_decoder_block3(det2, skip3)
        seg3 = self.seg_decoder_block3(seg2, skip3)
        det3, seg3 = self.cross_stitch3(det3, seg3)

        det4 = self.det_decoder_block4(det3, skip4)        
        seg4 = self.seg_decoder_block4(seg3, skip4)        
        det4, seg4 = self.cross_stitch4(det4, seg4)
        
        det5 = self.det_decoder_block5(det4)
        seg5 = self.seg_decoder_block5(seg4)
        det5, seg5 = self.cross_stitch5(det5, seg5)

        # head
        det5 = self.det_head(det5)
        seg5 = self.seg_head(seg5)
        poi = self.poi_head(poi)

        return det5, seg5, poi

class Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI(nn.Module):
    def __init__(self):
        super(Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI, self).__init__()

        # Encoder
        self.encoder = timm.create_model('maxvit_xlarge_tf_512.in21k_ft_in1k', pretrained=True, features_only=True) # Xlarge

        # DET Decoder
        self.det_decoder_block1 = DET_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block2 = DET_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block3 = DET_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block4 = DET_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block5 = DET_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)

        # SEG Decoder
        self.seg_decoder_block1 = SEG_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block2 = SEG_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block3 = SEG_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block4 = SEG_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block5 = SEG_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)
        
        # REC Decoder
        self.rec_decoder_block1 = REC_Skip_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block2 = REC_Skip_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block3 = REC_Skip_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block4 = REC_Skip_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block5 = REC_Skip_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)


        # POI Decoder   
        self.poi_decoder_block1 = nn.AdaptiveAvgPool2d(1)
        self.poi_decoder_block2 = LayerNorm2d(num_channels=1536, eps=1e-5, affine=True)
        self.poi_decoder_block3 = Flatten()
        self.poi_decoder_block4 = nn.Sequential(nn.Linear(in_features=1536, out_features=1536, bias=True), nn.Tanh())
        self.poi_decoder_block5 = nn.Dropout(p=0.2, inplace=False)

        # Head
        self.det_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.seg_head = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.rec_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.poi_head = nn.Linear(in_features=1536, out_features=2, bias=True)


        # Cross Stitch
        self.cross_stitch1 = CrossStitch_ConvRes_Unit(in_channels=256*2, out_channels=256)
        self.cross_stitch2 = CrossStitch_ConvRes_Unit(in_channels=128*2, out_channels=128)
        self.cross_stitch3 = CrossStitch_ConvRes_Unit(in_channels=64*2,  out_channels=64)
        self.cross_stitch4 = CrossStitch_ConvRes_Unit(in_channels=32*2,  out_channels=32)
        self.cross_stitch5 = CrossStitch_ConvRes_Unit(in_channels=16*2,  out_channels=16)        

        # Init
        self.initialize()


    def initialize(self):
        initialize_decoder(self.det_decoder_block1)
        initialize_decoder(self.det_decoder_block2)
        initialize_decoder(self.det_decoder_block3)
        initialize_decoder(self.det_decoder_block4)
        initialize_decoder(self.det_decoder_block5)
        initialize_decoder(self.seg_decoder_block1)
        initialize_decoder(self.seg_decoder_block2)
        initialize_decoder(self.seg_decoder_block3)
        initialize_decoder(self.seg_decoder_block4)
        initialize_decoder(self.seg_decoder_block5) 
        initialize_decoder(self.rec_decoder_block1)
        initialize_decoder(self.rec_decoder_block2)
        initialize_decoder(self.rec_decoder_block3)
        initialize_decoder(self.rec_decoder_block4)
        initialize_decoder(self.rec_decoder_block5)
        initialize_decoder(self.poi_decoder_block1)
        initialize_decoder(self.poi_decoder_block2)
        initialize_decoder(self.poi_decoder_block3)
        initialize_decoder(self.poi_decoder_block4)
        initialize_decoder(self.poi_decoder_block5)          
        initialize_head(self.det_head)
        initialize_head(self.seg_head)
        initialize_head(self.rec_head)
        initialize_head(self.poi_head)


    def forward(self, x):
        # encoder
        skip4, skip3, skip2, skip1, x = self.encoder(x)

        det1 = self.det_decoder_block1(x,   skip1)
        seg1 = self.seg_decoder_block1(x,   skip1)
        det1, seg1 = self.cross_stitch1(det1, seg1)

        det2 = self.det_decoder_block2(det1, skip2)
        seg2 = self.seg_decoder_block2(seg1, skip2)
        det2, seg2 = self.cross_stitch2(det2, seg2)

        det3 = self.det_decoder_block3(det2, skip3)
        seg3 = self.seg_decoder_block3(seg2, skip3)
        det3, seg3 = self.cross_stitch3(det3, seg3)

        det4 = self.det_decoder_block4(det3, skip4)        
        seg4 = self.seg_decoder_block4(seg3, skip4)        
        det4, seg4 = self.cross_stitch4(det4, seg4)
        
        det5 = self.det_decoder_block5(det4)
        seg5 = self.seg_decoder_block5(seg4)
        det5, seg5 = self.cross_stitch5(det5, seg5)

        # poi decoder
        poi = self.poi_decoder_block1(x)
        poi = self.poi_decoder_block2(poi)
        poi = self.poi_decoder_block3(poi)
        poi = self.poi_decoder_block4(poi)
        poi = self.poi_decoder_block5(poi)

        # rec decoder
        rec = self.rec_decoder_block1(x,   skip1)
        rec = self.rec_decoder_block2(rec, skip2)
        rec = self.rec_decoder_block3(rec, skip3)
        rec = self.rec_decoder_block4(rec, skip4)
        rec = self.rec_decoder_block5(rec)

        # head
        det5 = self.det_head(det5)
        seg5 = self.seg_head(seg5)
        rec  = self.rec_head(rec)
        poi  = self.poi_head(poi)

        return det5, seg5, rec, poi

class Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method(nn.Module):
    def __init__(self):
        super(Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method, self).__init__()

        # Encoder
        self.encoder = timm.create_model('maxvit_xlarge_tf_512.in21k_ft_in1k', pretrained=True, features_only=True) # Xlarge

        # DET Decoder
        self.det_decoder_block1 = DET_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block2 = DET_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block3 = DET_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block4 = DET_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block5 = DET_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)

        # SEG Decoder
        self.seg_decoder_block1 = SEG_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block2 = SEG_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block3 = SEG_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block4 = SEG_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block5 = SEG_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)
        
        # # REC Decoder
        # self.rec_decoder_block1 = REC_DecoderBlock(in_channels=1536,  out_channels=256, use_batchnorm=True, attention_type='scse')
        # self.rec_decoder_block2 = REC_DecoderBlock(in_channels=256,   out_channels=128, use_batchnorm=True, attention_type='scse')
        # self.rec_decoder_block3 = REC_DecoderBlock(in_channels=128,   out_channels=64,  use_batchnorm=True, attention_type='scse')
        # self.rec_decoder_block4 = REC_DecoderBlock(in_channels=64,    out_channels=32,  use_batchnorm=True, attention_type='scse')
        # self.rec_decoder_block5 = REC_DecoderBlock(in_channels=32,    out_channels=16,  use_batchnorm=True, attention_type='scse')

        # REC Decoder
        self.rec_decoder_block1 = REC_Skip_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block2 = REC_Skip_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block3 = REC_Skip_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block4 = REC_Skip_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block5 = REC_Skip_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)


        # POI Decoder   
        self.poi_decoder_block1 = nn.AdaptiveAvgPool2d(1)
        self.poi_decoder_block2 = LayerNorm2d(num_channels=1536, eps=1e-5, affine=True)
        self.poi_decoder_block3 = Flatten()
        self.poi_decoder_block4 = nn.Sequential(nn.Linear(in_features=1536, out_features=1536, bias=True), nn.Tanh())
        self.poi_decoder_block5 = nn.Dropout(p=0.2, inplace=False)

        # Head
        self.det_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.seg_head = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.rec_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.poi_head = nn.Linear(in_features=1536, out_features=2, bias=True)

        # Cross Stitch
        # self.cross_stitch1 = CrossStitch_Conv_Unit(in_channels=256*2, out_channels=256)
        # self.cross_stitch2 = CrossStitch_Conv_Unit(in_channels=128*2, out_channels=128)
        # self.cross_stitch3 = CrossStitch_Conv_Unit(in_channels=64*2,  out_channels=64)
        # self.cross_stitch4 = CrossStitch_Conv_Unit(in_channels=32*2,  out_channels=32)
        # self.cross_stitch5 = CrossStitch_Conv_Unit(in_channels=16*2,  out_channels=16)

        self.cross_stitch1 = CrossStitch_ConvRes_Unit(in_channels=256*2, out_channels=256)
        self.cross_stitch2 = CrossStitch_ConvRes_Unit(in_channels=128*2, out_channels=128)
        self.cross_stitch3 = CrossStitch_ConvRes_Unit(in_channels=64*2,  out_channels=64)
        self.cross_stitch4 = CrossStitch_ConvRes_Unit(in_channels=32*2,  out_channels=32)
        self.cross_stitch5 = CrossStitch_ConvRes_Unit(in_channels=16*2,  out_channels=16)        

        # Init
        self.initialize()

    def initialize(self):
        initialize_decoder(self.det_decoder_block1)
        initialize_decoder(self.det_decoder_block2)
        initialize_decoder(self.det_decoder_block3)
        initialize_decoder(self.det_decoder_block4)
        initialize_decoder(self.det_decoder_block5)
        initialize_decoder(self.seg_decoder_block1)
        initialize_decoder(self.seg_decoder_block2)
        initialize_decoder(self.seg_decoder_block3)
        initialize_decoder(self.seg_decoder_block4)
        initialize_decoder(self.seg_decoder_block5) 
        initialize_decoder(self.rec_decoder_block1)
        initialize_decoder(self.rec_decoder_block2)
        initialize_decoder(self.rec_decoder_block3)
        initialize_decoder(self.rec_decoder_block4)
        initialize_decoder(self.rec_decoder_block5)
        initialize_decoder(self.poi_decoder_block1)
        initialize_decoder(self.poi_decoder_block2)
        initialize_decoder(self.poi_decoder_block3)
        initialize_decoder(self.poi_decoder_block4)
        initialize_decoder(self.poi_decoder_block5)          
        initialize_head(self.det_head)
        initialize_head(self.seg_head)
        initialize_head(self.rec_head)
        initialize_head(self.poi_head)

    def shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(self.encoder.parameters())

    def task_specific_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return chain(
            self.det_decoder_block1.parameters(),
            self.det_decoder_block2.parameters(),
            self.det_decoder_block3.parameters(),
            self.det_decoder_block4.parameters(),
            self.det_decoder_block5.parameters(),
            self.seg_decoder_block1.parameters(),
            self.seg_decoder_block2.parameters(),
            self.seg_decoder_block3.parameters(),
            self.seg_decoder_block4.parameters(),
            self.seg_decoder_block5.parameters(),
            self.rec_decoder_block1.parameters(),
            self.rec_decoder_block2.parameters(),
            self.rec_decoder_block3.parameters(),
            self.rec_decoder_block4.parameters(),
            self.rec_decoder_block5.parameters(),
            self.poi_decoder_block1.parameters(),
            self.poi_decoder_block2.parameters(),
            self.poi_decoder_block3.parameters(),
            self.poi_decoder_block4.parameters(),
            self.poi_decoder_block5.parameters(),
            self.det_head.parameters(),
            self.seg_head.parameters(),
            self.rec_head.parameters(),
            self.poi_head.parameters(),
            self.cross_stitch1.parameters(),
            self.cross_stitch2.parameters(),
            self.cross_stitch3.parameters(),
            self.cross_stitch4.parameters(),
            self.cross_stitch5.parameters(),
        )

    def last_shared_parameters(self) -> Iterator[nn.parameter.Parameter]:
        return self.encoder.stages_3.blocks[1].parameters()



    def forward(self, x):
        # encoder
        skip4, skip3, skip2, skip1, x = self.encoder(x)

        det1 = self.det_decoder_block1(x,   skip1)
        seg1 = self.seg_decoder_block1(x,   skip1)
        det1, seg1 = self.cross_stitch1(det1, seg1)

        det2 = self.det_decoder_block2(det1, skip2)
        seg2 = self.seg_decoder_block2(seg1, skip2)
        det2, seg2 = self.cross_stitch2(det2, seg2)

        det3 = self.det_decoder_block3(det2, skip3)
        seg3 = self.seg_decoder_block3(seg2, skip3)
        det3, seg3 = self.cross_stitch3(det3, seg3)

        det4 = self.det_decoder_block4(det3, skip4)        
        seg4 = self.seg_decoder_block4(seg3, skip4)        
        det4, seg4 = self.cross_stitch4(det4, seg4)
        
        det5 = self.det_decoder_block5(det4)
        seg5 = self.seg_decoder_block5(seg4)
        det5, seg5 = self.cross_stitch5(det5, seg5)

        # poi decoder
        poi = self.poi_decoder_block1(x)
        poi = self.poi_decoder_block2(poi)
        poi = self.poi_decoder_block3(poi)
        poi = self.poi_decoder_block4(poi)
        poi = self.poi_decoder_block5(poi)

        # # rec decoder
        # rec = self.rec_decoder_block1(x)
        # rec = self.rec_decoder_block2(rec)
        # rec = self.rec_decoder_block3(rec)
        # rec = self.rec_decoder_block4(rec)        
        # rec = self.rec_decoder_block5(rec)

        # rec decoder
        rec = self.rec_decoder_block1(x,   skip1)
        rec = self.rec_decoder_block2(rec, skip2)
        rec = self.rec_decoder_block3(rec, skip3)
        rec = self.rec_decoder_block4(rec, skip4)
        rec = self.rec_decoder_block5(rec)

        # head
        det5 = self.det_head(det5)
        seg5 = self.seg_head(seg5)
        rec  = self.rec_head(rec)
        poi  = self.poi_head(poi)

        # x is last shared feature
        return det5, seg5, rec, poi, x






# Final Finetuning
class Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Original(nn.Module):
    def __init__(self):
        super(Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Original, self).__init__()

        # Encoder
        self.encoder = timm.create_model('maxvit_xlarge_tf_512.in21k_ft_in1k', pretrained=True, features_only=True) # Xlarge

        # DET Decoder
        self.det_decoder_block1 = DET_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block2 = DET_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block3 = DET_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block4 = DET_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block5 = DET_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse')

        # SEG Decoder
        self.seg_decoder_block1 = SEG_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block2 = SEG_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block3 = SEG_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block4 = SEG_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.seg_decoder_block5 = SEG_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse')
        
        # REC Decoder
        self.rec_decoder_block1 = REC_DecoderBlock(in_channels=1536,  out_channels=256, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block2 = REC_DecoderBlock(in_channels=256,   out_channels=128, use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block3 = REC_DecoderBlock(in_channels=128,   out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block4 = REC_DecoderBlock(in_channels=64,    out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.rec_decoder_block5 = REC_DecoderBlock(in_channels=32,    out_channels=16,  use_batchnorm=True, attention_type='scse')

        # POI Decoder   
        self.poi_decoder_block1 = nn.AdaptiveAvgPool2d(1)
        self.poi_decoder_block2 = LayerNorm2d(num_channels=1536, eps=1e-5, affine=True)
        self.poi_decoder_block3 = Flatten()
        self.poi_decoder_block4 = nn.Sequential(nn.Linear(in_features=1536, out_features=1536, bias=True), nn.Tanh())
        self.poi_decoder_block5 = nn.Dropout(p=0.2, inplace=False)

        # Head
        self.det_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.seg_head = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.rec_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)
        self.poi_head = nn.Linear(in_features=1536, out_features=2, bias=True)

        self.cross_stitch1 = CrossStitch_Conv_Unit(in_channels=256*2, out_channels=256)
        self.cross_stitch2 = CrossStitch_Conv_Unit(in_channels=128*2, out_channels=128)
        self.cross_stitch3 = CrossStitch_Conv_Unit(in_channels=64*2,  out_channels=64)
        self.cross_stitch4 = CrossStitch_Conv_Unit(in_channels=32*2,  out_channels=32)
        self.cross_stitch5 = CrossStitch_Conv_Unit(in_channels=16*2,  out_channels=16)

        # Init
        self.initialize()


    def initialize(self):
        initialize_decoder(self.det_decoder_block1)
        initialize_decoder(self.det_decoder_block2)
        initialize_decoder(self.det_decoder_block3)
        initialize_decoder(self.det_decoder_block4)
        initialize_decoder(self.det_decoder_block5)
        initialize_decoder(self.seg_decoder_block1)
        initialize_decoder(self.seg_decoder_block2)
        initialize_decoder(self.seg_decoder_block3)
        initialize_decoder(self.seg_decoder_block4)
        initialize_decoder(self.seg_decoder_block5) 
        initialize_decoder(self.rec_decoder_block1)
        initialize_decoder(self.rec_decoder_block2)
        initialize_decoder(self.rec_decoder_block3)
        initialize_decoder(self.rec_decoder_block4)
        initialize_decoder(self.rec_decoder_block5)
        initialize_decoder(self.poi_decoder_block1)
        initialize_decoder(self.poi_decoder_block2)
        initialize_decoder(self.poi_decoder_block3)
        initialize_decoder(self.poi_decoder_block4)
        initialize_decoder(self.poi_decoder_block5)          
        initialize_head(self.det_head)
        initialize_head(self.seg_head)
        initialize_head(self.rec_head)
        initialize_head(self.poi_head)


    def forward(self, x):
        # encoder
        skip4, skip3, skip2, skip1, x = self.encoder(x)

        det1 = self.det_decoder_block1(x,    skip1)
        seg1 = self.seg_decoder_block1(x,    skip1)
        det1, seg1 = self.cross_stitch1(det1, seg1)

        det2 = self.det_decoder_block2(det1, skip2)
        seg2 = self.seg_decoder_block2(seg1, skip2)
        det2, seg2 = self.cross_stitch2(det2, seg2)

        det3 = self.det_decoder_block3(det2, skip3)
        seg3 = self.seg_decoder_block3(seg2, skip3)
        det3, seg3 = self.cross_stitch3(det3, seg3)

        det4 = self.det_decoder_block4(det3, skip4)        
        seg4 = self.seg_decoder_block4(seg3, skip4)        
        det4, seg4 = self.cross_stitch4(det4, seg4)
        
        det5 = self.det_decoder_block5(det4)
        seg5 = self.seg_decoder_block5(seg4)
        det5, seg5 = self.cross_stitch5(det5, seg5)

        # poi decoder
        poi = self.poi_decoder_block1(x)
        poi = self.poi_decoder_block2(poi)
        poi = self.poi_decoder_block3(poi)
        poi = self.poi_decoder_block4(poi)
        poi = self.poi_decoder_block5(poi)

        # rec decoder
        rec = self.rec_decoder_block1(x)
        rec = self.rec_decoder_block2(rec)
        rec = self.rec_decoder_block3(rec)
        rec = self.rec_decoder_block4(rec)        
        rec = self.rec_decoder_block5(rec)

        # head
        det5 = self.det_head(det5)
        seg5 = self.seg_head(seg5)
        rec  = self.rec_head(rec)
        poi  = self.poi_head(poi)

        return det5, seg5, rec, poi


class Cell_MaxViT_UNet_DET(nn.Module):
    def __init__(self):
        super(Cell_MaxViT_UNet_DET, self).__init__()

        # Encoder
        self.encoder = timm.create_model('maxvit_xlarge_tf_512.in21k_ft_in1k', pretrained=True, features_only=True) # Xlarge

        # DET Decoder
        self.det_decoder_block1 = DET_DecoderBlock(in_channels=1536, skip_channels=768, out_channels=256, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block2 = DET_DecoderBlock(in_channels=256,  skip_channels=384, out_channels=128, use_batchnorm=True, attention_type='scse')
        self.det_decoder_block3 = DET_DecoderBlock(in_channels=128,  skip_channels=192, out_channels=64,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block4 = DET_DecoderBlock(in_channels=64,   skip_channels=192, out_channels=32,  use_batchnorm=True, attention_type='scse')
        self.det_decoder_block5 = DET_DecoderBlock(in_channels=32,   skip_channels=0,   out_channels=16,  use_batchnorm=True, attention_type='scse', last=True)

        # Head
        self.det_head = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1)

        # Init
        self.initialize()
        self.load_state_dict()


    def initialize(self):
        initialize_decoder(self.det_decoder_block1)
        initialize_decoder(self.det_decoder_block2)
        initialize_decoder(self.det_decoder_block3)
        initialize_decoder(self.det_decoder_block4)
        initialize_decoder(self.det_decoder_block5)
        initialize_head(self.det_head)

    def load_state_dict(self):
        print("Load State Dict...!")
        checkpoint      = torch.load('/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230801_Cell_MaxViT_UNet_MTL_Cross_Conv_Point_Xlarge_CLAHE_Focal/epoch_105_checkpoint.pth')
        pretrained_dict = checkpoint['model_state_dict']
        model_dict      = self.state_dict()
        
        print("이전 weight = ", model_dict['encoder.stem.conv1.weight'][0])
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        print("이후 weight = ", model_dict['encoder.stem.conv1.weight'][0])
        super().load_state_dict(model_dict)


    def forward(self, x):
        # encoder
        skip4, skip3, skip2, skip1, x = self.encoder(x)

        det1 = self.det_decoder_block1(x,   skip1)

        det2 = self.det_decoder_block2(det1, skip2)

        det3 = self.det_decoder_block3(det2, skip3)

        det4 = self.det_decoder_block4(det3, skip4)        
        
        det5 = self.det_decoder_block5(det4)

        # head
        det5 = self.det_head(det5)

        return det5
