import os 
import sys
import math
import utils
import torch
import metrics
import datetime
import numpy as np
import pandas as pd
import albumentations as A
import torch.nn.functional as F

from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

# Setting...!
fn_denorm  = lambda x: (x * 0.5) + 0.5
fn_tonumpy = lambda x: x.cpu().detach().numpy()

'''
Declares the training and validation Loops.
'''





# ----------------------------------------------------------------------------------------------
def train_cell_loop_fn(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader))
    
    for step, batch_data in enumerate(epoch_iterator):

        image, cell_gt, gt_point_path = batch_data
        image, cell_gt = image.to(device), cell_gt.to(device)

        logit = model(image)

        # print(image.dtype, image.max(), image.min())                            torch.float32 tensor(1., device='cuda:0') tensor(-0.9529, device='cuda:0')
        # print(tissue_segmap.dtype, tissue_segmap.max(), tissue_segmap.min())    torch.float32 tensor(1., device='cuda:0') tensor(0., device='cuda:0')
        # print(logit.dtype, logit.max(), logit.min())                            torch.float32 tensor(2.7825, device='cuda:0', grad_fn=<MaxBackward1>) tensor(-5.2060, device='cuda:0', grad_fn=<MinBackward1>)
        # print(cell_gt.dtype, cell_gt.max(), cell_gt.min())                      torch.int64 tensor(255, device='cuda:0') tensor(1, device='cuda:0')

        loss = criterion(logit, cell_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(key='train_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def valid_cell_loop_fn(valid_loader, model, criterion, device, epoch, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader))

    pred_all = [] # must be 1d list or array
    gt_all   = [] # must be 1d list or array
    for step, batch_data in enumerate(epoch_iterator):

        image, cell_gt, gt_point_path = batch_data
        image, cell_gt = image.to(device), cell_gt.to(device)

        logit = model(image)
        # logit 이 tuple, List 형태인지 check
        loss  = criterion(logit, cell_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='valid_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps), (valid_loss=%2.5f)" % (epoch, step, len(valid_loader), loss_value))

        # Post-processing
        if isinstance(logit, tuple) or isinstance(logit, list):
            logit = logit[1]

        preds = logit.softmax(dim=1).detach().cpu()
        gts   = cell_gt.detach().cpu()
        
        # Metric Calculation
        score = metrics.multiclass_dice_score(y_true=gts.squeeze().numpy(), y_pred=preds.squeeze().numpy())
        metric_logger.update(key='valid_dice_score', value=score, n=image.shape[0])

        resize_preds    = F.interpolate(preds, size=(1024, 1024), mode='bilinear', align_corners=True).squeeze().numpy()
        predicted_cells = metrics.find_cells(resize_preds)

        try:
            gt_cells = pd.read_csv(gt_point_path[0], header=None).values
        except:
            gt_cells = np.zeros(shape=(0, 3))

        pred_all.append(predicted_cells)
        gt_all.append(gt_cells)

    # PNG Save
    image_png = (fn_tonumpy(fn_denorm(image)).squeeze()*255).astype('uint8') # (1, 3, 1024, 1024)
    preds_png = preds.squeeze().argmax(dim=0).numpy() # (1024, 1024)
    image_png[1][preds_png==1] = 255
    image_png[2][preds_png==2] = 255
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred.png', image_png.transpose(1, 2, 0))
    
    image_png = (fn_tonumpy(fn_denorm(image)).squeeze()*255).astype('uint8') # (1, 3, 1024, 1024)
    gts_png   = gts.squeeze().argmax(dim=0).numpy() # (1024, 1024)
    image_png[1][gts_png==1] = 255
    image_png[2][gts_png==2] = 255
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt.png', image_png.transpose(1, 2, 0))

    # Metric Calculation
    mf1 = metrics.mf1_metric(pred_all, gt_all)
    metric_logger.update(key='valid_mF1', value=mf1, n=image.shape[0])

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def test_cell_loop_fn(test_loader, model, criterion, device, save_dir, checkpoint_dir, tta):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))

    # TTA를 위한 변환 정의
    n_augmentations = 5
    tta_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    ])

    save_dict = dict()
    preds = [] # must be 1d list or array
    gts   = [] # must be 1d list or array
    for step, batch_data in enumerate(epoch_iterator):
        image, target, path = batch_data
        image, target = image.to(device), target.to(device)

        if tta:
            logit = model(image)
            loss  = criterion(logit, target)
            loss_value = loss.item()

        else:
            all_outputs = []
            for _ in range(n_augmentations):
                augmented_images = tta_transforms(image=image)['image']
                outputs = model(augmented_images)
                all_outputs.append(outputs)
            logit = torch.stack(all_outputs).mean(0)
            loss  = criterion(logit, target)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='test_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Testing: (%d / %d Steps), (test_loss=%2.5f)" % (step, len(test_loader), loss_value))

        # post-processing
        preds.append(logit.squeeze().detach().cpu().numpy())
        gts.append(target.squeeze().detach().cpu().numpy())

    # Metric Calculation
    preds = np.array(preds)
    gts   = np.array(gts)

    # pearsonr, spearmanr, kendalltau
    plcc, srocc, krocc, overall = metric_correlation_coefficient(preds, gts)
    metric_logger.update(key='PLCC',    value=plcc.item(), n=image.shape[0])
    metric_logger.update(key='SROCC',   value=srocc.item(), n=image.shape[0])
    metric_logger.update(key='KROCC',   value=krocc.item(), n=image.shape[0])
    metric_logger.update(key='Overall', value=overall.item(), n=image.shape[0])

    # r2 score
    r2 = r2_score(gts, preds)
    # print("gts == ", gts)
    # print("gts shape == ", gts.shape)
    # print("preds == ", preds)
    # print("preds shape == ", preds.shape)    
    metric_logger.update(key='R2', value=r2.item(), n=1)

    # NPZ
    save_dict['plcc']    = plcc.item()
    save_dict['srocc']   = srocc.item()
    save_dict['krocc']   = krocc.item()
    save_dict['overall'] = overall.item()
    save_dict['r2']      = r2.item()

    # save save_dict CSV (for excel)
    df = pd.DataFrame(save_dict, index=[0])
    df.to_csv(checkpoint_dir + '/test_results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".csv", index=False)
    print("Save test results to " + checkpoint_dir + '/test_results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".csv")
    
    return {k: round(v, 7) for k, v in metric_logger.average().items()}



# ----------------------------------------------------------------------------------------------
def train_cell_mtl_det_seg_rec_loop_fn(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader))
    
    for step, batch_data in enumerate(epoch_iterator):

        image, tissue_segmap, cell_gt, gt_point_path = batch_data
        image, tissue_segmap, cell_gt = image.to(device), tissue_segmap.to(device), cell_gt.to(device)

        logit_det, logit_seg, logit_rec = model(image)
        loss = criterion(logit_det, logit_seg, logit_rec, cell_gt, tissue_segmap, image)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(key='train_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def valid_cell_mtl_det_seg_rec_loop_fn(valid_loader, model, criterion, device, epoch, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader))

    pred_all = [] # must be 1d list or array
    gt_all   = [] # must be 1d list or array
    for step, batch_data in enumerate(epoch_iterator):

        image, tissue_segmap, cell_gt, gt_point_path = batch_data
        image, tissue_segmap, cell_gt = image.to(device), tissue_segmap.to(device), cell_gt.to(device)

        logit_det, logit_seg, logit_rec = model(image)
        loss = criterion(logit_det, logit_seg, logit_rec, cell_gt, tissue_segmap, image)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='valid_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps), (valid_loss=%2.5f)" % (epoch, step, len(valid_loader), loss_value))

        # post-processing
        preds = logit_det.softmax(dim=1).detach().cpu()
        gts   = cell_gt.detach().cpu()
        
        # Metric Calculation
        score = metrics.multiclass_dice_score(y_true=gts.squeeze().numpy(), y_pred=preds.squeeze().numpy())
        metric_logger.update(key='valid_dice_score', value=score, n=image.shape[0])

        resize_preds = F.interpolate(preds, size=(1024, 1024), mode='bilinear', align_corners=True).squeeze().numpy()
        predicted_cells = metrics.find_cells(resize_preds)

        try:
            gt_cells = pd.read_csv(gt_point_path[0], header=None).values
        except:
            gt_cells = np.zeros(shape=(0, 3))

        pred_all.append(predicted_cells)
        gt_all.append(gt_cells)

    # PNG Save
    image_png = (fn_tonumpy(fn_denorm(image)).squeeze()*255).astype('uint8') # (1, 3, 1024, 1024)
    preds_png = preds.squeeze().argmax(dim=0).numpy() # (1024, 1024)
    image_png[0][preds_png==1] = 0
    image_png[1][preds_png==1] = 255
    image_png[2][preds_png==1] = 0

    image_png[0][preds_png==2] = 0
    image_png[1][preds_png==2] = 0
    image_png[2][preds_png==2] = 255
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred.png', image_png.transpose(1, 2, 0))
    
    image_png = (fn_tonumpy(fn_denorm(image)).squeeze()*255).astype('uint8') # (1, 3, 1024, 1024)
    gts_png   = gts.squeeze().argmax(dim=0).numpy() # (1024, 1024)
    image_png[0][gts_png==1] = 0
    image_png[1][gts_png==1] = 255
    image_png[2][gts_png==1] = 0

    image_png[0][gts_png==2] = 0
    image_png[1][gts_png==2] = 0
    image_png[2][gts_png==2] = 255
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt.png', image_png.transpose(1, 2, 0))

    # Metric Calculation
    mf1 = metrics.mf1_metric(pred_all, gt_all)
    metric_logger.update(key='valid_mF1', value=mf1, n=image.shape[0])

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def test_cell_mtl_det_seg_rec_loop_fn(test_loader, model, criterion, device, save_dir, checkpoint_dir, tta):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))

    # TTA를 위한 변환 정의
    n_augmentations = 5
    tta_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    ])

    save_dict = dict()
    preds = [] # must be 1d list or array
    gts   = [] # must be 1d list or array
    for step, batch_data in enumerate(epoch_iterator):
        image, target, path = batch_data
        image, target = image.to(device), target.to(device)

        if tta:
            logit = model(image)
            loss  = criterion(logit, target)
            loss_value = loss.item()

        else:
            all_outputs = []
            for _ in range(n_augmentations):
                augmented_images = tta_transforms(image=image)['image']
                outputs = model(augmented_images)
                all_outputs.append(outputs)
            logit = torch.stack(all_outputs).mean(0)
            loss  = criterion(logit, target)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='test_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Testing: (%d / %d Steps), (test_loss=%2.5f)" % (step, len(test_loader), loss_value))

        # post-processing
        preds.append(logit.squeeze().detach().cpu().numpy())
        gts.append(target.squeeze().detach().cpu().numpy())

    # Metric Calculation
    preds = np.array(preds)
    gts   = np.array(gts)

    # pearsonr, spearmanr, kendalltau
    plcc, srocc, krocc, overall = metric_correlation_coefficient(preds, gts)
    metric_logger.update(key='PLCC',    value=plcc.item(), n=image.shape[0])
    metric_logger.update(key='SROCC',   value=srocc.item(), n=image.shape[0])
    metric_logger.update(key='KROCC',   value=krocc.item(), n=image.shape[0])
    metric_logger.update(key='Overall', value=overall.item(), n=image.shape[0])

    # r2 score
    r2 = r2_score(gts, preds)
    # print("gts == ", gts)
    # print("gts shape == ", gts.shape)
    # print("preds == ", preds)
    # print("preds shape == ", preds.shape)    
    metric_logger.update(key='R2', value=r2.item(), n=1)

    # NPZ
    save_dict['plcc']    = plcc.item()
    save_dict['srocc']   = srocc.item()
    save_dict['krocc']   = krocc.item()
    save_dict['overall'] = overall.item()
    save_dict['r2']      = r2.item()

    # save save_dict CSV (for excel)
    df = pd.DataFrame(save_dict, index=[0])
    df.to_csv(checkpoint_dir + '/test_results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".csv", index=False)
    print("Save test results to " + checkpoint_dir + '/test_results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".csv")
    
    return {k: round(v, 7) for k, v in metric_logger.average().items()}



# ----------------------------------------------------------------------------------------------
def train_cell_mtl_det_seg_poi_loop_fn(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader))
    
    for step, batch_data in enumerate(epoch_iterator):

        image, tissue_segmap, cell_gt, point_gt, gt_point_path = batch_data
        image, tissue_segmap, cell_gt, point_gt = image.to(device), tissue_segmap.to(device), cell_gt.to(device), point_gt.to(device)

        logit_det, logit_seg, logit_poi = model(image)
        loss = criterion(logit_det, logit_seg, logit_poi, cell_gt, tissue_segmap, point_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(key='train_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def valid_cell_mtl_det_seg_poi_loop_fn(valid_loader, model, criterion, device, epoch, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader))

    pred_all = [] # must be 1d list or array
    gt_all   = [] # must be 1d list or array
    for step, batch_data in enumerate(epoch_iterator):
        
        image, tissue_segmap, cell_gt, point_gt, gt_point_path = batch_data
        image, tissue_segmap, cell_gt, point_gt = image.to(device), tissue_segmap.to(device), cell_gt.to(device), point_gt.to(device)

        logit_det, logit_seg, logit_poi = model(image)
        loss = criterion(logit_det, logit_seg, logit_poi, cell_gt, tissue_segmap, point_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='valid_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps), (valid_loss=%2.5f)" % (epoch, step, len(valid_loader), loss_value))

        # post-processing
        preds = logit_det.softmax(dim=1).detach().cpu()
        gts   = cell_gt.detach().cpu()
        
        # Metric Calculation
        score = metrics.multiclass_dice_score(y_true=gts.squeeze().numpy(), y_pred=preds.squeeze().numpy())
        metric_logger.update(key='valid_dice_score', value=score, n=image.shape[0])

        resize_preds = F.interpolate(preds, size=(1024, 1024), mode='bilinear', align_corners=True).squeeze().numpy()
        predicted_cells = metrics.find_cells(resize_preds)

        try:
            gt_cells = pd.read_csv(gt_point_path[0], header=None).values
        except:
            gt_cells = np.zeros(shape=(0, 3))

        pred_all.append(predicted_cells)
        gt_all.append(gt_cells)

    # PNG Save
    image_png = (fn_tonumpy(fn_denorm(image)).squeeze()*255).astype('uint8') # (1, 3, 1024, 1024)
    preds_png = preds.squeeze().argmax(dim=0).numpy() # (1024, 1024)
    image_png[0][preds_png==1] = 0
    image_png[1][preds_png==1] = 255
    image_png[2][preds_png==1] = 0

    image_png[0][preds_png==2] = 0
    image_png[1][preds_png==2] = 0
    image_png[2][preds_png==2] = 255
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred.png', image_png.transpose(1, 2, 0))
    
    image_png = (fn_tonumpy(fn_denorm(image)).squeeze()*255).astype('uint8') # (1, 3, 1024, 1024)
    gts_png   = gts.squeeze().argmax(dim=0).numpy() # (1024, 1024)
    image_png[0][gts_png==1] = 0
    image_png[1][gts_png==1] = 255
    image_png[2][gts_png==1] = 0

    image_png[0][gts_png==2] = 0
    image_png[1][gts_png==2] = 0
    image_png[2][gts_png==2] = 255
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt.png', image_png.transpose(1, 2, 0))

    # Metric Calculation
    mf1 = metrics.mf1_metric(pred_all, gt_all)
    metric_logger.update(key='valid_mF1', value=mf1, n=image.shape[0])

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def test_cell_mtl_det_seg_poi_loop_fn(test_loader, model, criterion, device, save_dir, checkpoint_dir, tta):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))

    # TTA를 위한 변환 정의
    n_augmentations = 5
    tta_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    ])

    save_dict = dict()
    preds = [] # must be 1d list or array
    gts   = [] # must be 1d list or array
    for step, batch_data in enumerate(epoch_iterator):
        image, target, path = batch_data
        image, target = image.to(device), target.to(device)

        if tta:
            logit = model(image)
            loss  = criterion(logit, target)
            loss_value = loss.item()

        else:
            all_outputs = []
            for _ in range(n_augmentations):
                augmented_images = tta_transforms(image=image)['image']
                outputs = model(augmented_images)
                all_outputs.append(outputs)
            logit = torch.stack(all_outputs).mean(0)
            loss  = criterion(logit, target)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='test_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Testing: (%d / %d Steps), (test_loss=%2.5f)" % (step, len(test_loader), loss_value))

        # post-processing
        preds.append(logit.squeeze().detach().cpu().numpy())
        gts.append(target.squeeze().detach().cpu().numpy())

    # Metric Calculation
    preds = np.array(preds)
    gts   = np.array(gts)

    # pearsonr, spearmanr, kendalltau
    plcc, srocc, krocc, overall = metric_correlation_coefficient(preds, gts)
    metric_logger.update(key='PLCC',    value=plcc.item(), n=image.shape[0])
    metric_logger.update(key='SROCC',   value=srocc.item(), n=image.shape[0])
    metric_logger.update(key='KROCC',   value=krocc.item(), n=image.shape[0])
    metric_logger.update(key='Overall', value=overall.item(), n=image.shape[0])

    # r2 score
    r2 = r2_score(gts, preds)
    # print("gts == ", gts)
    # print("gts shape == ", gts.shape)
    # print("preds == ", preds)
    # print("preds shape == ", preds.shape)    
    metric_logger.update(key='R2', value=r2.item(), n=1)

    # NPZ
    save_dict['plcc']    = plcc.item()
    save_dict['srocc']   = srocc.item()
    save_dict['krocc']   = krocc.item()
    save_dict['overall'] = overall.item()
    save_dict['r2']      = r2.item()

    # save save_dict CSV (for excel)
    df = pd.DataFrame(save_dict, index=[0])
    df.to_csv(checkpoint_dir + '/test_results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".csv", index=False)
    print("Save test results to " + checkpoint_dir + '/test_results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".csv")
    
    return {k: round(v, 7) for k, v in metric_logger.average().items()}







# ----------------------------------------------------------------------------------------------
def train_cell_mtl_det_seg_rec_poi_loop_fn(train_loader, model, criterion, optimizer, device, epoch):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader))
    
    for step, batch_data in enumerate(epoch_iterator):

        image, tissue_segmap, cell_gt, point_gt, gt_point_path = batch_data
        image, tissue_segmap, cell_gt, point_gt = image.to(device), tissue_segmap.to(device), cell_gt.to(device), point_gt.to(device)

        logit_det, logit_seg, logit_rec, logit_poi = model(image)
        loss = criterion(logit_det, logit_seg, logit_rec, logit_poi, cell_gt, tissue_segmap, image, point_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(key='train_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def valid_cell_mtl_det_seg_rec_poi_loop_fn(valid_loader, model, criterion, device, epoch, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader))

    pred_all = [] # must be 1d list or array
    gt_all   = [] # must be 1d list or array
    for step, batch_data in enumerate(epoch_iterator):
        
        image, tissue_segmap, cell_gt, point_gt, gt_point_path = batch_data
        image, tissue_segmap, cell_gt, point_gt = image.to(device), tissue_segmap.to(device), cell_gt.to(device), point_gt.to(device)

        logit_det, logit_seg, logit_rec, logit_poi = model(image)
        loss = criterion(logit_det, logit_seg, logit_rec, logit_poi, cell_gt, tissue_segmap, image, point_gt)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='valid_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps), (valid_loss=%2.5f)" % (epoch, step, len(valid_loader), loss_value))

        # post-processing
        preds = logit_det.softmax(dim=1).detach().cpu()
        gts   = cell_gt.detach().cpu()
        
        # Metric Calculation
        score = metrics.multiclass_dice_score(y_true=gts.squeeze().numpy(), y_pred=preds.squeeze().numpy())
        metric_logger.update(key='valid_dice_score', value=score, n=image.shape[0])

        resize_preds = F.interpolate(preds, size=(1024, 1024), mode='bilinear', align_corners=True).squeeze().numpy()
        predicted_cells = metrics.find_cells(resize_preds)

        try:
            gt_cells = pd.read_csv(gt_point_path[0], header=None).values
        except:
            gt_cells = np.zeros(shape=(0, 3))

        pred_all.append(predicted_cells)
        gt_all.append(gt_cells)

    # PNG Save
    image_png = (fn_tonumpy(fn_denorm(image)).squeeze()*255).astype('uint8') # (1, 3, 1024, 1024)
    preds_png = preds.squeeze().argmax(dim=0).numpy() # (1024, 1024)
    image_png[0][preds_png==1] = 0
    image_png[1][preds_png==1] = 255
    image_png[2][preds_png==1] = 0

    image_png[0][preds_png==2] = 0
    image_png[1][preds_png==2] = 0
    image_png[2][preds_png==2] = 255
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred.png', image_png.transpose(1, 2, 0))
    
    image_png = (fn_tonumpy(fn_denorm(image)).squeeze()*255).astype('uint8') # (1, 3, 1024, 1024)
    gts_png   = gts.squeeze().argmax(dim=0).numpy() # (1024, 1024)
    image_png[0][gts_png==1] = 0
    image_png[1][gts_png==1] = 255
    image_png[2][gts_png==1] = 0

    image_png[0][gts_png==2] = 0
    image_png[1][gts_png==2] = 0
    image_png[2][gts_png==2] = 255
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt.png', image_png.transpose(1, 2, 0))

    # Metric Calculation
    mf1 = metrics.mf1_metric(pred_all, gt_all)
    metric_logger.update(key='valid_mF1', value=mf1, n=image.shape[0])

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def test_cell_mtl_det_seg_rec_poi_loop_fn(test_loader, model, criterion, device, save_dir, checkpoint_dir, tta):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))

    # TTA를 위한 변환 정의
    n_augmentations = 5
    tta_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    ])

    save_dict = dict()
    preds = [] # must be 1d list or array
    gts   = [] # must be 1d list or array
    for step, batch_data in enumerate(epoch_iterator):
        image, target, path = batch_data
        image, target = image.to(device), target.to(device)

        if tta:
            logit = model(image)
            loss  = criterion(logit, target)
            loss_value = loss.item()

        else:
            all_outputs = []
            for _ in range(n_augmentations):
                augmented_images = tta_transforms(image=image)['image']
                outputs = model(augmented_images)
                all_outputs.append(outputs)
            logit = torch.stack(all_outputs).mean(0)
            loss  = criterion(logit, target)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='test_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Testing: (%d / %d Steps), (test_loss=%2.5f)" % (step, len(test_loader), loss_value))

        # post-processing
        preds.append(logit.squeeze().detach().cpu().numpy())
        gts.append(target.squeeze().detach().cpu().numpy())

    # Metric Calculation
    preds = np.array(preds)
    gts   = np.array(gts)

    # pearsonr, spearmanr, kendalltau
    plcc, srocc, krocc, overall = metric_correlation_coefficient(preds, gts)
    metric_logger.update(key='PLCC',    value=plcc.item(), n=image.shape[0])
    metric_logger.update(key='SROCC',   value=srocc.item(), n=image.shape[0])
    metric_logger.update(key='KROCC',   value=krocc.item(), n=image.shape[0])
    metric_logger.update(key='Overall', value=overall.item(), n=image.shape[0])

    # r2 score
    r2 = r2_score(gts, preds)
    # print("gts == ", gts)
    # print("gts shape == ", gts.shape)
    # print("preds == ", preds)
    # print("preds shape == ", preds.shape)    
    metric_logger.update(key='R2', value=r2.item(), n=1)

    # NPZ
    save_dict['plcc']    = plcc.item()
    save_dict['srocc']   = srocc.item()
    save_dict['krocc']   = krocc.item()
    save_dict['overall'] = overall.item()
    save_dict['r2']      = r2.item()

    # save save_dict CSV (for excel)
    df = pd.DataFrame(save_dict, index=[0])
    df.to_csv(checkpoint_dir + '/test_results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".csv", index=False)
    print("Save test results to " + checkpoint_dir + '/test_results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".csv")
    
    return {k: round(v, 7) for k, v in metric_logger.average().items()}







# ----------------------------------------------------------------------------------------------
def train_cell_mtl_det_seg_rec_poi_method_loop_fn(train_loader, model, criterion, optimizer, device, epoch, method):
    model.train()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(train_loader))
    
    for step, batch_data in enumerate(epoch_iterator):

        image, tissue_segmap, cell_gt, point_gt, gt_point_path = batch_data
        image, tissue_segmap, cell_gt, point_gt = image.to(device), tissue_segmap.to(device), cell_gt.to(device), point_gt.to(device)

        logit_det, logit_seg, logit_rec, logit_poi, feat = model(image)
        losses = criterion(logit_det, logit_seg, logit_rec, logit_poi, cell_gt, tissue_segmap, image, point_gt)

        optimizer.zero_grad()        
        loss, extra_outputs = method.backward(losses=losses, 
                                              shared_parameters=list(model.shared_parameters()), 
                                              task_specific_parameters=list(model.task_specific_parameters()), 
                                              last_shared_parameters=list(model.last_shared_parameters()), 
                                              representation=feat)
        if loss is not None:
            loss_value = loss.item()

        else:
            loss = torch.sum(losses)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))            

        optimizer.step()
        metric_logger.update(key='train_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Training: Epochs %d (%d / %d Steps), (train_loss=%2.5f)" % (epoch, step, len(train_loader), loss_value))

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def valid_cell_mtl_det_seg_rec_poi_method_loop_fn(valid_loader, model, criterion, device, epoch, save_dir):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(valid_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(valid_loader))

    pred_all = [] # must be 1d list or array
    gt_all   = [] # must be 1d list or array
    for step, batch_data in enumerate(epoch_iterator):
        
        image, tissue_segmap, cell_gt, point_gt, gt_point_path = batch_data
        image, tissue_segmap, cell_gt, point_gt = image.to(device), tissue_segmap.to(device), cell_gt.to(device), point_gt.to(device)

        logit_det, logit_seg, logit_rec, logit_poi, _ = model(image)
        losses = criterion(logit_det, logit_seg, logit_rec, logit_poi, cell_gt, tissue_segmap, image, point_gt)
        loss = torch.sum(losses)
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='valid_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Validating: Epochs %d (%d / %d Steps), (valid_loss=%2.5f)" % (epoch, step, len(valid_loader), loss_value))

        # post-processing
        preds = logit_det.softmax(dim=1).detach().cpu()
        gts   = cell_gt.detach().cpu()
        
        # Metric Calculation
        score = metrics.multiclass_dice_score(y_true=gts.squeeze().numpy(), y_pred=preds.squeeze().numpy())
        metric_logger.update(key='valid_dice_score', value=score, n=image.shape[0])

        resize_preds = F.interpolate(preds, size=(1024, 1024), mode='bilinear', align_corners=True).squeeze().numpy()
        predicted_cells = metrics.find_cells(resize_preds)

        try:
            gt_cells = pd.read_csv(gt_point_path[0], header=None).values
        except:
            gt_cells = np.zeros(shape=(0, 3))

        pred_all.append(predicted_cells)
        gt_all.append(gt_cells)

    # PNG Save
    image_png = (fn_tonumpy(fn_denorm(image)).squeeze()*255).astype('uint8') # (1, 3, 1024, 1024)
    preds_png = preds.squeeze().argmax(dim=0).numpy() # (1024, 1024)
    image_png[0][preds_png==1] = 0
    image_png[1][preds_png==1] = 255
    image_png[2][preds_png==1] = 0

    image_png[0][preds_png==2] = 0
    image_png[1][preds_png==2] = 0
    image_png[2][preds_png==2] = 255
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_pred.png', image_png.transpose(1, 2, 0))
    
    image_png = (fn_tonumpy(fn_denorm(image)).squeeze()*255).astype('uint8') # (1, 3, 1024, 1024)
    gts_png   = gts.squeeze().argmax(dim=0).numpy() # (1024, 1024)
    image_png[0][gts_png==1] = 0
    image_png[1][gts_png==1] = 255
    image_png[2][gts_png==1] = 0

    image_png[0][gts_png==2] = 0
    image_png[1][gts_png==2] = 0
    image_png[2][gts_png==2] = 255
    plt.imsave(save_dir+'/epoch_'+str(epoch)+'_gt.png', image_png.transpose(1, 2, 0))

    # Metric Calculation
    mf1 = metrics.mf1_metric(pred_all, gt_all)
    metric_logger.update(key='valid_mF1', value=mf1, n=image.shape[0])

    return {k: round(v, 7) for k, v in metric_logger.average().items()}

# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def test_cell_mtl_det_seg_rec_poi_method_loop_fn(test_loader, model, criterion, device, save_dir, checkpoint_dir, tta):
    model.eval()
    metric_logger  = utils.AverageMeter()
    epoch_iterator = tqdm(test_loader, desc="Validating (X / X Steps) (loss=X.X)", dynamic_ncols=True, total=len(test_loader))

    # TTA를 위한 변환 정의
    n_augmentations = 5
    tta_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
    ])

    save_dict = dict()
    preds = [] # must be 1d list or array
    gts   = [] # must be 1d list or array
    for step, batch_data in enumerate(epoch_iterator):
        image, target, path = batch_data
        image, target = image.to(device), target.to(device)

        if tta:
            logit = model(image)
            loss  = criterion(logit, target)
            loss_value = loss.item()

        else:
            all_outputs = []
            for _ in range(n_augmentations):
                augmented_images = tta_transforms(image=image)['image']
                outputs = model(augmented_images)
                all_outputs.append(outputs)
            logit = torch.stack(all_outputs).mean(0)
            loss  = criterion(logit, target)
            loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))

        metric_logger.update(key='test_loss', value=loss_value, n=image.shape[0])
        epoch_iterator.set_description("Testing: (%d / %d Steps), (test_loss=%2.5f)" % (step, len(test_loader), loss_value))

        # post-processing
        preds.append(logit.squeeze().detach().cpu().numpy())
        gts.append(target.squeeze().detach().cpu().numpy())

    # Metric Calculation
    preds = np.array(preds)
    gts   = np.array(gts)

    # pearsonr, spearmanr, kendalltau
    plcc, srocc, krocc, overall = metric_correlation_coefficient(preds, gts)
    metric_logger.update(key='PLCC',    value=plcc.item(), n=image.shape[0])
    metric_logger.update(key='SROCC',   value=srocc.item(), n=image.shape[0])
    metric_logger.update(key='KROCC',   value=krocc.item(), n=image.shape[0])
    metric_logger.update(key='Overall', value=overall.item(), n=image.shape[0])

    # r2 score
    r2 = r2_score(gts, preds)
    # print("gts == ", gts)
    # print("gts shape == ", gts.shape)
    # print("preds == ", preds)
    # print("preds shape == ", preds.shape)    
    metric_logger.update(key='R2', value=r2.item(), n=1)

    # NPZ
    save_dict['plcc']    = plcc.item()
    save_dict['srocc']   = srocc.item()
    save_dict['krocc']   = krocc.item()
    save_dict['overall'] = overall.item()
    save_dict['r2']      = r2.item()

    # save save_dict CSV (for excel)
    df = pd.DataFrame(save_dict, index=[0])
    df.to_csv(checkpoint_dir + '/test_results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".csv", index=False)
    print("Save test results to " + checkpoint_dir + '/test_results_' + datetime.datetime.now().strftime("%y%m%d_%H%M") + ".csv")
    
    return {k: round(v, 7) for k, v in metric_logger.average().items()}



