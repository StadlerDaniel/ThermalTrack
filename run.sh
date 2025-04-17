#!/usr/bin/env bash
GPU_ID=0
#!!! SET THE DATASET ROOT
TMOT_DATASET_ROOT=
SEQUENCES_DIR=${TMOT_DATASET_ROOT}/images/val
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
OUTPUT_DIR=${SCRIPT_DIR}/output5

mkdir -p ${OUTPUT_DIR}

# write dataset root into detection config
echo -e "data_root = '${TMOT_DATASET_ROOT}'\n$(cat ${SCRIPT_DIR}/models/detection_config.py)" > ${SCRIPT_DIR}/models/detection_config.py
# generate detections
CUDA_VISIBLE_DEVICES=${GPU_ID} python mmyolo/tools/test.py ${SCRIPT_DIR}/models/detection_config.py ${SCRIPT_DIR}/models/detector_weights.pth --out ${OUTPUT_DIR}/detections.pkl --work-dir ${OUTPUT_DIR}
# format detections
python format_detections.py ${OUTPUT_DIR}/detections.pkl ${SEQUENCES_DIR}
# runtime measurement detector
CUDA_VISIBLE_DEVICES=${GPU_ID} python mmyolo/tools/analysis_tools/benchmark.py models/detection_config.py models/detector_weights.pth --max-iter 1000 --work-dir ${OUTPUT_DIR}
# extract features
CUDA_VISIBLE_DEVICES=${GPU_ID} python extract_features.py ${OUTPUT_DIR}/detections ${SEQUENCES_DIR} ${SCRIPT_DIR}/models/reid_config.yml ${SCRIPT_DIR}/models/reid_weights.pth ${OUTPUT_DIR}/reid_runtime.txt
# tracking
python track.py ${OUTPUT_DIR}/detections_w_features ${OUTPUT_DIR}
