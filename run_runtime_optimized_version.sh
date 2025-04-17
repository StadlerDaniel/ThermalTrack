#!/usr/bin/env bash
GPU_ID=0
#!!! SET THE DATASET ROOT
TMOT_DATASET_ROOT=
SEQUENCES_DIR=${TMOT_DATASET_ROOT}/images/val
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_DIR=${SCRIPT_DIR}/output_runtime_optimized

mkdir -p ${OUTPUT_DIR}

# write dataset root into detection config
echo -e "data_root = '${TMOT_DATASET_ROOT}'\n$(cat ${SCRIPT_DIR}/models/detection_config_trt.py)" > ${SCRIPT_DIR}/models/detection_config_trt.py
# generate detections
CUDA_VISIBLE_DEVICES=${GPU_ID} python generate_detections_trt.py ${SCRIPT_DIR}/mmyolo/configs/deploy/detection_tensorrt-fp16_static-1504x1888.py ${SCRIPT_DIR}/models/detection_config_trt.py ${SCRIPT_DIR}/models/detector_bs1_compute8.9_end2end.engine ${OUTPUT_DIR}/detections_trt 
# extract features
CUDA_VISIBLE_DEVICES=${GPU_ID} python extract_features_osnet_trt.py ${OUTPUT_DIR}/detections_trt ${SEQUENCES_DIR} ${SCRIPT_DIR}/models/reid_osnet_bs_dynamic_compute8.9.trt
# tracking
python track.py ${OUTPUT_DIR}/detections_trt_w_trt_osnet_features ${OUTPUT_DIR} --s1_min_score 0.5 --s2_max_score 0.5 --s1_metrics diou,app_cos --s2_metrics diou,app_cos --s1_weights 1,4 --s2_weights 1,4 --s1_dist_thresh 2.3 --s2_dist_thresh 1.8 --filter_min_len 0
