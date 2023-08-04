import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation


class Tissue_Segformer(nn.Module):
    def __init__(self):
        super(Tissue_Segformer, self).__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        self.model.decode_head.classifier = nn.Conv2d(768, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        output = F.interpolate(input=logits, size=(512, 512), mode='bilinear', align_corners=True)
        return output


class Cell_Segformer(nn.Module):
    def __init__(self):
        super(Cell_Segformer, self).__init__()

        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        self.model.decode_head.classifier = nn.Conv2d(768, 3, kernel_size=(1, 1), stride=(1, 1), bias=True)

    def forward(self, x):
        logits = self.model(pixel_values=x).logits
        output = F.interpolate(input=logits, size=(512, 512), mode='bilinear', align_corners=True)
        return output
    


class Tissue_Cell_Segformer(nn.Module):
    def __init__(self):
        super(Tissue_Cell_Segformer, self).__init__()
        self.t_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        self.t_model.decode_head.classifier = nn.Conv2d(768, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)

        self.c_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
        self.c_model.segformer.encoder.patch_embeddings[0].proj = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3), bias=True) # input channel 4
        self.c_model.decode_head.classifier = nn.Conv2d(768, 3, kernel_size=(1, 1), stride=(1, 1), bias=True)
        
        # initialize tissue model weight
        self.load_tissue_model_weight()


    def load_tissue_model_weight(self):
        checkpoint = torch.load('/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230720_Tissue_Segformer_Segmap_Save_Cancer/epoch_262_checkpoint.pth')
        checkpoint = {k.replace('model.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        self.t_model.load_state_dict(checkpoint)
        print("Loading Tissue Model Weight...!")

    def forward(self, tissue, cell, position):
        # batch wise
        output_list = []
        for idx in range(cell.size(0)):
            t = tissue[idx].unsqueeze(0)
            c = cell[idx].unsqueeze(0)
            p = position[idx].unsqueeze(0)

            # tissue (extract cell)
            with torch.no_grad():
                self.t_model.eval()
                t_seg = self.t_model(pixel_values=t).logits
                t_seg = F.interpolate(input=t_seg, size=(512, 512), mode='bilinear', align_corners=True)
                t_seg = torch.sigmoid(t_seg)
                
                points = torch.where(p == 1)
                y_min, y_max = torch.min(points[2]), torch.max(points[2])
                x_min, x_max = torch.min(points[3]), torch.max(points[3])
                t_seg = t_seg[:, :, y_min:y_max+1, x_min:x_max+1]
                t_seg = F.interpolate(input=t_seg, size=(512, 512), mode='bilinear', align_corners=True) # B, C, 512, 512

            # cell
            x = torch.cat([c, t_seg], dim=1) # B, 4, 512, 512
            logits = self.c_model(pixel_values=x).logits
            output = F.interpolate(input=logits, size=(512, 512), mode='bilinear', align_corners=True).squeeze()
            output_list.append(output)

        return torch.stack(output_list, dim=0)
