#!/usr/bin/env bash
# Run from the repository root with the PENET environment activated.
export CUDA_VISIBLE_DEVICES=0
echo "Testing!!!!"
MODEL_NAME="PE-NET_PredCls_NoReweight"
python -m tools.relation_test_net \
        --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
        MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
        MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
        MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork \
        TEST.IMS_PER_BATCH 1 DTYPE "float32" \
        GLOVE_DIR ./Datasets/ \
        MODEL.PRETRAINED_DETECTOR_CKPT ./Datasets/model_final.pth \
        MODEL.WEIGHT ./checkpoints/${MODEL_NAME}/model_final.pth \
        OUTPUT_DIR ./checkpoints/${MODEL_NAME} \
        TEST.ALLOW_LOAD_FROM_CACHE False
