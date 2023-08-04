#!/bin/sh

: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_more' \
--batch-size 2 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_Cross_Conv_Point' \
--loss 'cell_maxvit_unet_mtl_more_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230801_Cell_MaxViT_UNet_MTL_Cross_Conv_Point_Xlarge_CLAHE_Focal' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230801_Cell_MaxViT_UNet_MTL_Cross_Conv_Point_Xlarge_CLAHE_Focal' \
--memo 'image 512x512, 230801_Cell_MaxViT_UNet_MTL_Cross_Conv_Point_Xlarge_CLAHE_Focal, using det, seg, rec, CLAHE Focal loss'
END


: <<'END'
CUDA_VISIBLE_DEVICES=6 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_more' \
--batch-size 2 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_Cross_ResALL' \
--loss 'cell_maxvit_unet_mtl_more_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230801_Cell_MaxViT_UNet_MTL_Cross_ResALL_CLAHE_Focal_remove_point' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230801_Cell_MaxViT_UNet_MTL_Cross_ResALL_CLAHE_Focal_remove_point' \
--memo 'image 512x512, 230801_Cell_MaxViT_UNet_MTL_Cross_ResALL_CLAHE_Focal, using det, seg, rec, CLAHE Focal loss'
END

: <<'END'
CUDA_VISIBLE_DEVICES=7 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_more' \
--batch-size 3 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_Cross_ResALL_V2' \
--loss 'cell_maxvit_unet_mtl_more_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230801_Cell_MaxViT_UNet_MTL_Cross_ResALL_V2_CLAHE_Focal' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230801_Cell_MaxViT_UNet_MTL_Cross_ResALL_V2_CLAHE_Focal' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_Cross_ResALL_V2, using det, seg, rec, CLAHE Focal loss'
END



# NEW

: <<'END'
CUDA_VISIBLE_DEVICES=0 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_poi' \
--batch-size 3 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_POI' \
--loss 'cell_det_seg_poi_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_POI' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_POI' \
--memo 'image 512x512, 230803_Cell_MaxViT_UNet_MTL_DET_SEG_POI, using det, seg, rec, CLAHE Focal loss'
END


: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 3 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI' \
--loss 'cell_det_seg_rec_poi_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_ConvRes' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_ConvRes' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI, using det, seg, rec, CLAHE Focal loss'
END


: <<'END'
CUDA_VISIBLE_DEVICES=6 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 3 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI' \
--loss 'cell_det_seg_rec_poi_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_SkipREC' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_SkipREC' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI, using det, seg, rec, CLAHE Focal loss'
END


: <<'END'
CUDA_VISIBLE_DEVICES=4 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 3 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI' \
--loss 'cell_det_seg_rec_poi_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI, using det, seg, rec, CLAHE Focal loss'
END

: <<'END'
CUDA_VISIBLE_DEVICES=5 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 3 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI' \
--loss 'cell_det_seg_rec_poi_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_loss_weight' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_loss_weight' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI, using det, seg, rec, CLAHE Focal loss'
END


: <<'END'
CUDA_VISIBLE_DEVICES=7 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 1 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method' \
--loss 'cell_det_seg_rec_poi_method_loss' \
--method 'nashmtl' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method_Nashmtl' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method_Nashmtl' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method, using det, seg, rec, CLAHE Focal loss'
END


: <<'END'
CUDA_VISIBLE_DEVICES=2 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 1 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method' \
--loss 'cell_det_seg_rec_poi_method_loss' \
--method 'cagrad' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method_CAGrad' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method_CAGrad' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method, using det, seg, rec, CLAHE Focal loss'
END

: <<'END'
CUDA_VISIBLE_DEVICES=3 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 2 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method' \
--loss 'cell_det_seg_rec_poi_method_loss' \
--method 'pcgrad' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method_PCGrad' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method_PCGrad' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method, using det, seg, rec, CLAHE Focal loss'
END



: <<'END'
CUDA_VISIBLE_DEVICES=2 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 3 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI' \
--loss 'cell_det_seg_rec_poi_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_loss_weight' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_loss_weight' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI, using det, seg, rec, CLAHE Focal loss' \
--resume '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_loss_weight/epoch_107_checkpoint.pth'
END


: <<'END'
CUDA_VISIBLE_DEVICES=1 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 1 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method' \
--loss 'cell_det_seg_rec_poi_method_loss' \
--method 'cagrad' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method_CAGrad_Final' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method_CAGrad_Final' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_Method, using det, seg, rec, CLAHE Focal loss'
END



: <<'END'
CUDA_VISIBLE_DEVICES=3 python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 3 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI' \
--loss 'cell_det_seg_rec_poi_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_loss_weight_Final' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_loss_weight_Final' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI, using det, seg, rec, CLAHE Focal loss'
END






# Finetune
: <<'END'
CUDA_VISIBLE_DEVICES=4 python -W ignore train.py \
--dataset 'ocelot_cell_segmentation' \
--batch-size 4 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_DET' \
--loss 'cell_dice_ce_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-5 \
--multi-gpu-mode 'Single' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230804_Cell_MaxViT_UNet_DET_Final' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230804_Cell_MaxViT_UNet_DET_Final' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI, using det, seg, rec, CLAHE Focal loss'
END



: <<'END'
CUDA_VISIBLE_DEVICES=0,1,2,5,6,7  python -W ignore train.py \
--dataset 'ocelot_cell_mtl_det_seg_rec_poi' \
--batch-size 24 \
--num_workers 9 \
--model 'Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI' \
--loss 'cell_det_seg_rec_poi_loss' \
--optimizer 'adamw' \
--scheduler 'poly_lr' \
--epochs 500 \
--lr 1e-4 \
--multi-gpu-mode 'DataParallel' \
--checkpoint-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/checkpoints/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_loss_weight_Real' \
--save-dir '/workspace/sunggu/0.Challenge/MICCAI2023_OCELOT/predictions/230803_Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI_loss_weight_Real' \
--memo 'image 512x512, Cell_MaxViT_UNet_MTL_DET_SEG_REC_POI, using det, seg, rec, CLAHE Focal loss'
END