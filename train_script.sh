#!/bin/bash
MODEL_NAME='din50'

TARGET_SET='webpage'
TARGET_IMAGE='/path/to/target/dataset/image'
TARGET_FIXATION='/path/to/target/dataset/fixationmap'
SOURCE_SET='salicon'
SOURCE_IMAGE='/path/to/source/dataset/image'
SOURCE_FIXATION='/path/to/source/dataset/fixationmap'

OUTPUT_DIR='logs/'${SOURCE_SET}'_'${MODEL_NAME}'_'${TARGET_SET}
PRETRAINED_EPOCH=3
PRETRAINED="./model/pretrained_DINet.pth"
LR=0.00005
BATCH_SIZE=5 # 1 3 5 for 1-, 5-, and 10-shot
NUM_SHOTS=10
TRIALS=1
export CUDA_VISIBLE_DEVICES=0
# conventional training
python train.py \
	--batch_size 10 \
	--epochs 10 \
	--lr ${LR} \
	--lr_decay_epoch 3 \
	--lr_coef 0.2 \
	--weight_decay 1e-4 \
	--model ${MODEL_NAME} \
	--out_dir "${OUTPUT_DIR}/baseline/" \
	--train_img_dir "${SOURCE_IMAGE}/" \
	--train_gt_dir "${SOURCE_FIXATION}/" \
	--val_img_dir "${TARGET_IMAGE}/" \
	--val_gt_dir "${TARGET_FIXATION}/" \
	--image_size 320 480 \
	--tr_fxt_size 480 640 \
	--val_fxt_size 1360 768
# n-reference learning
python train_nref.py \
	--batch_size 10 \
	--epochs 10 \
	--lr ${LR} \
	--lr_decay_epoch 3 \
	--lr_coef 0.2 \
	--weight_decay 1e-4 \
	--model ${MODEL_NAME} \
	--out_dir "${OUTPUT_DIR}/${NUM_SHOTS}shot_${TRIALS}/tr_ref_val" \
	--train_img_dir "${SOURCE_IMAGE}/" \
	--train_gt_dir "${SOURCE_FIXATION}/" \
	--val_img_dir "${TARGET_IMAGE}/" \
	--val_gt_dir "${TARGET_FIXATION}/" \
	--num_shots ${NUM_SHOTS} \
	--ref_batch_size ${BATCH_SIZE} \
	--image_size 320 480 \
	--tr_fxt_size 480 640 \
	--val_fxt_size 1360 768 \
	--ref_fxt_size 1360 768
# n-shot learning
python train_nshot.py \
	--batch_size 10 \
	--epochs 10 \
	--lr ${LR} \
	--lr_decay_epoch 3 \
	--lr_coef 0.2 \
	--weight_decay 1e-4 \
	--model ${MODEL_NAME} \
	--out_dir "${OUTPUT_DIR}/${NUM_SHOTS}shot_${TRIALS}/tr_ft_val" \
	--train_img_dir "${SOURCE_IMAGE}/" \
	--train_gt_dir "${SOURCE_FIXATION}/" \
	--val_img_dir "${TARGET_IMAGE}/" \
	--val_gt_dir "${TARGET_FIXATION}/" \
	--pretrainedModel ${PRETRAINED} \
	--split_file "${OUTPUT_DIR}/${NUM_SHOTS}shot_${TRIALS}/tr_ref_val/split_data.npz" \
	--num_shots ${NUM_SHOTS} \
	--image_size 320 480 \
	--val_fxt_size 1360 768
# n-reference fine-tuning
python train_nshot.py \
	--batch_size 10 \
	--epochs 10 \
	--lr ${LR} \
	--lr_decay_epoch 3 \
	--lr_coef 0.2 \
	--weight_decay 1e-4 \
	--model ${MODEL_NAME} \
	--out_dir "${OUTPUT_DIR}/${NUM_SHOTS}shot_${TRIALS}/tr_ref_ft_val" \
	--train_img_dir "${SOURCE_IMAGE}/" \
	--train_gt_dir "${SOURCE_FIXATION}/" \
	--val_img_dir "${TARGET_IMAGE}/" \
	--val_gt_dir "${TARGET_FIXATION}/" \
	--pretrainedModel "${OUTPUT_DIR}/${NUM_SHOTS}shot_${TRIALS}/tr_ref_val/snapshots/model_ep${PRETRAINED_EPOCH}.pth" \
	--pretrainedModel_head "${OUTPUT_DIR}/${NUM_SHOTS}shot_${TRIALS}/tr_ref_val/snapshots/model_head_ep${PRETRAINED_EPOCH}.pth" \
	--split_file "${OUTPUT_DIR}/${NUM_SHOTS}shot_${TRIALS}/tr_ref_val/split_data.npz" \
	--num_shots ${NUM_SHOTS} \
	--image_size 320 480 \
	--val_fxt_size 1360 768