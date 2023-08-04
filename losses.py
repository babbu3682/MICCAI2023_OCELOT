import torch
import torch.nn.functional as F



def dice_loss(y_true, y_pred, eps=1e-7):
    # y_true: 실제 타깃. 크기는 (배치 크기, 클래스 수, 높이, 너비).
    # y_pred: 네트워크의 출력. 크기는 (배치 크기, 클래스 수, 높이, 너비).
    mask = torch.sum(y_true, dim=(0, 2, 3)) > 0

    # 각 클래스에 대해 별도로 Dice 손실을 계산
    intersection = torch.sum(y_true * y_pred, dim=(0, 2, 3))
    cardinality  = torch.sum(y_true + y_pred, dim=(0, 2, 3))

    dice_score = (2. * intersection) / cardinality.clamp_min(eps)
    dice_loss = 1 - dice_score

    return (dice_loss * mask.float()).mean()



def focal_loss(y_pred, y_true, gamma=2.0, alpha=0.25):
    logpt  = F.binary_cross_entropy(y_pred, y_true, reduction="none")
    # compute the loss
    focal_term = (1.0 - torch.exp(-logpt)).pow(gamma)
    loss = focal_term * logpt
    loss *= alpha*y_true + (1-alpha)*(1-y_true)
    return loss.mean()


def softmax_focal_loss(y_pred, y_true, gamma=2.0):
    loss = F.cross_entropy(y_pred, y_true, reduction="none", weight=torch.tensor([0.6, 0.2, 0.2]).cuda())
    # compute the loss
    focal_term = (1.0 - torch.exp(-loss)).pow(gamma)
    loss = focal_term * loss
    return loss.mean()




class Cell_Dice_CE_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_ce   = F.cross_entropy
        self.loss_dice = dice_loss

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()

        ce_loss   = self.loss_ce(input=y_pred.log_softmax(dim=1), target=y_true)
        dice_loss = self.loss_dice(y_pred=y_pred.softmax(dim=1), y_true=y_true)
        return ce_loss + dice_loss

class Cell_Dice_CE_MSE_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_ce   = F.cross_entropy
        self.loss_dice = dice_loss
        self.loss_mse  = F.mse_loss

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        
        ce_loss   = self.loss_ce(input=y_pred.log_softmax(dim=1), target=y_true)
        dice_loss = self.loss_dice(y_pred=y_pred.softmax(dim=1), y_true=y_true)
        mse_loss  = self.loss_mse(input=y_pred.softmax(dim=1), target=y_true)
        return ce_loss + dice_loss + mse_loss

class Cell_DET_SEG_REC_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_det_ce = softmax_focal_loss
        self.loss_seg_ce = focal_loss
        self.loss_dice   = dice_loss
        self.loss_l1     = F.l1_loss

    def forward(self, logit_det, logit_seg, logit_rec, cell_gt, tissue_segmap, image):
        assert logit_det.size() == cell_gt.size()
        assert logit_seg.size() == tissue_segmap.size()
        assert logit_rec.size() == image.size()
        
        det_ce_loss   = self.loss_det_ce(y_pred=logit_det.log_softmax(dim=1), y_true=cell_gt)
        det_dice_loss = self.loss_dice(y_pred=logit_det.softmax(dim=1), y_true=cell_gt)

        seg_ce_loss   = self.loss_seg_ce(y_pred=torch.sigmoid(logit_seg), y_true=tissue_segmap)
        seg_dice_loss = self.loss_dice(y_pred=torch.sigmoid(logit_seg), y_true=tissue_segmap)

        rec_dice_loss = self.loss_l1(logit_rec, image)

        return det_ce_loss + det_dice_loss + seg_ce_loss + seg_dice_loss + rec_dice_loss

class Cell_DET_SEG_POI_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_det_ce = softmax_focal_loss
        self.loss_seg_ce = focal_loss
        self.loss_dice   = dice_loss
        self.loss_l2     = F.mse_loss

    def forward(self, logit_det, logit_seg, logit_poi, cell_gt, tissue_segmap, poi_gt):
        assert logit_det.size() == cell_gt.size()
        assert logit_seg.size() == tissue_segmap.size()
        
        det_ce_loss   = self.loss_det_ce(y_pred=logit_det.log_softmax(dim=1), y_true=cell_gt)
        det_dice_loss = self.loss_dice(y_pred=logit_det.softmax(dim=1), y_true=cell_gt)

        seg_ce_loss   = self.loss_seg_ce(y_pred=torch.sigmoid(logit_seg), y_true=tissue_segmap)
        seg_dice_loss = self.loss_dice(y_pred=torch.sigmoid(logit_seg), y_true=tissue_segmap)

        poi_l2_loss   = self.loss_l2(logit_poi, poi_gt)

        return det_ce_loss + det_dice_loss + seg_ce_loss + seg_dice_loss + poi_l2_loss

class Cell_DET_SEG_REC_POI_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_det_ce = softmax_focal_loss
        self.loss_seg_ce = focal_loss
        self.loss_dice   = dice_loss
        self.loss_l1     = F.l1_loss
        self.loss_l2     = F.mse_loss

    def forward(self, logit_det, logit_seg, logit_rec, logit_poi, cell_gt, tissue_segmap, image, poi_gt):
        assert logit_det.size() == cell_gt.size()
        assert logit_seg.size() == tissue_segmap.size()
        assert logit_rec.size() == image.size()
        
        det_ce_loss   = self.loss_det_ce(y_pred=logit_det.log_softmax(dim=1), y_true=cell_gt)
        det_dice_loss = self.loss_dice(y_pred=logit_det.softmax(dim=1), y_true=cell_gt)

        seg_ce_loss   = self.loss_seg_ce(y_pred=torch.sigmoid(logit_seg), y_true=tissue_segmap)
        seg_dice_loss = self.loss_dice(y_pred=torch.sigmoid(logit_seg), y_true=tissue_segmap)

        rec_l1_loss = self.loss_l1(logit_rec, image)

        poi_l2_loss = self.loss_l2(logit_poi, poi_gt)

        return det_ce_loss + det_dice_loss + seg_ce_loss + seg_dice_loss + rec_l1_loss + 10.0*poi_l2_loss

class Cell_DET_SEG_REC_POI_Method_Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_det_ce = softmax_focal_loss
        self.loss_seg_ce = focal_loss
        self.loss_dice   = dice_loss
        self.loss_l1     = F.l1_loss
        self.loss_l2     = F.mse_loss

    def forward(self, logit_det, logit_seg, logit_rec, logit_poi, cell_gt, tissue_segmap, image, poi_gt):
        assert logit_det.size() == cell_gt.size()
        assert logit_seg.size() == tissue_segmap.size()
        assert logit_rec.size() == image.size()
        
        det_ce_loss   = self.loss_det_ce(y_pred=logit_det.log_softmax(dim=1), y_true=cell_gt)
        det_dice_loss = self.loss_dice(y_pred=logit_det.softmax(dim=1), y_true=cell_gt)

        seg_ce_loss   = self.loss_seg_ce(y_pred=torch.sigmoid(logit_seg), y_true=tissue_segmap)
        seg_dice_loss = self.loss_dice(y_pred=torch.sigmoid(logit_seg), y_true=tissue_segmap)

        rec_l1_loss = self.loss_l1(logit_rec, image)

        poi_l2_loss = self.loss_l2(logit_poi, poi_gt)

        return torch.stack([det_ce_loss+det_dice_loss, seg_ce_loss+seg_dice_loss, rec_l1_loss, poi_l2_loss])


def get_loss(name):
    if name == "cell_dice_ce_loss":
        return Cell_Dice_CE_Loss()

    elif name == 'cell_dice_ce_mse_loss':
        return Cell_Dice_CE_MSE_Loss()

    elif name == 'cell_det_seg_rec_loss':
        return Cell_DET_SEG_REC_Loss()    
    
    elif name == 'cell_det_seg_poi_loss':
        return Cell_DET_SEG_POI_Loss()        

    elif name == 'cell_det_seg_rec_poi_loss':
        return Cell_DET_SEG_REC_POI_Loss()           

    elif name == 'cell_det_seg_rec_poi_method_loss':
        return Cell_DET_SEG_REC_POI_Method_Loss()

    else:
        raise NotImplementedError