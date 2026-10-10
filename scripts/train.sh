#!/bin/bash
# Vanilla PENET PredCls, without class-balanced loss / reweighting.
# Run from the repository root with the PENET environment activated.
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME='PE-NET_PredCls_NoReweight'
mkdir -p ./checkpoints/${MODEL_NAME}/
cp ./maskrcnn_benchmark/modeling/roi_heads/relation_head/loss.py ./checkpoints/${MODEL_NAME}/

python -m tools.relation_train_net \
  --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
  MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
  MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
  MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
  MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
  MODEL.ROI_RELATION_HEAD.LABEL_SMOOTHING_LOSS False \
  DTYPE "float32" \
  SOLVER.IMS_PER_BATCH 8 TEST.IMS_PER_BATCH 1 \
  SOLVER.MAX_ITER 60000 SOLVER.BASE_LR 1e-3 \
  SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
  MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 512 \
  SOLVER.STEPS "(28000, 48000)" SOLVER.VAL_PERIOD 30000 \
  SOLVER.CHECKPOINT_PERIOD 30000 GLOVE_DIR ./Datasets/ \
  MODEL.PRETRAINED_DETECTOR_CKPT ./Datasets/model_final.pth \
  OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
  SOLVER.PRE_VAL False \
  SOLVER.GRAD_NORM_CLIP 5.0
