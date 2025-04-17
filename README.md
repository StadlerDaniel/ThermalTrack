# ThermalTrack: A Strong Baseline for Multi-Person Tracking in Thermal Infrared Imagery (CVPRW 2025)

This repository contains the source code of the submission from Fraunhofer IOSB to the Thermal MOT Challenge 2025, held in conjunction with the [21st Workhsop on Perception Beyond the Visible Spectrum](https://pbvs-workshop.github.io/) at CVPR 2025. It is also the official code base of the corresponding paper "A Strong Baseline for Multi-Person Tracking in Thermal Infrared Imagery".

## Abstract
Multi-person tracking is a crucial component in many computer vision solutions for autonomous driving or surveillance
related tasks. While extensive research exists in the visible spectrum, the applicability of established multiperson
tracking approaches to thermal infrared images is largely unexplored, despite its high relevance for practical
applications. This work investigates the importance of commonly used tracking modules for detection, motion
modeling, person re-identification, and association in the thermal domain. On the basis of our findings, we develop
a strong multi-person tracker for thermal imagery, which significantly outperforms the baseline method of the novel
Thermal MOT dataset (+15.6 MOTA, +22.5 IDF1). With comprehensive experiments, differences to tracking on data
in the visual spectrum are revealed, and our single tracking components are explored in detail. Moreover, tackling
the limitations of many existing methods for real-time applications, we develop a runtime-optimized version of
our tracker, which runs at 81 FPS, while still achieving state-of-the-art results.

## Results on Thermal MOT validation set

|      Version      | HOTA | MOTA | MOTP | Rec. | Prec. | IDF1 | IDRe | IDPr | FPS  |
|-------------------|------|------|------|------|-------|------|------|------|------|
| Runtime-optimized | 68.8 | 80.3 | 85.5 | 86.3 | 94.2  | 84.3 | 80.8 | 88.2 | 80.6 |
| Accuracy-focused  | 70.2 | 82.2 | 85.6 | 88.1 | 94.3  | 86.3 | 83.4 | 89.3 | 6.1  |

## Installation

Follow the commands below to install all required packages.

```bash
# create anaconda environment
conda create --name pbvs_challenge python=3.8 -y
conda activate pbvs_challenge
# install MMYOLO
cd mmyolo
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install -r requirements/mminstall.txt
pip install -r requirements/albu.txt
mim install -v -e .
pip install tensorboard
pip install albumentations==1.3.1
# install additional package for SOLIDER-REID
pip install yacs
# install additional package for TrackEval
pip install numpy==1.23.4
```

## Download dataset and SOLIDER-REID model weights

- Download the thermal MOT dataset from the official [repo](https://github.com/wassimea/thermalMOT).
- Download the SOLIDER-REID model weights from [here](https://owncloud.fraunhofer.de/index.php/s/GViKyw3rPw7oHAv) and save them as `models/reid_weights.pth`.

## Generate challenge results of the accuracy-focused version

Pleas set `TMOT_DATASET_ROOT` and `GPU_ID` (optional) in `run.sh` accordingly.
The provided script `run.sh` performs the following tasks:
1. Write dataset root into detection config.
2. Generate detections and save them in `output/detections.pkl`. COCO AP evaluation is performed and the results are saved in a folder `YYYYMMDD_HHMMSS/` named after the current date and time. Moreover, the detection config file is saved in the `output/` folder. 
3. Format detections: One detection file `seqXX.pkl` per sequence will be saved in `output/detections/`.
4. Perform runtime measurement of detector: Log will be saved in `output/benchmark.log`.
5. Extract features with SOLIDER-REID model. Features will be saved per sequence together with the detections in `output/detections_w_features/`. The runtime will be measured and written in `output/reid_runtime.txt`
6. Tracking is performed with the loaded detections and features. Tracking results are saved as `seqXX.txt` in `output/tracking/`. Tracking evaluation is performed and runtime is measured. The evaluation results can be found in `output/tracking/results.txt` and `output/tracking/pedestrian_summary.txt`. The measured runtime is saved in `output/tracking/tracker_runtime.txt`. The config file of the tracker is saved as `output/tracking/tracking.yaml`.

The generated files in `output/` should match the ones in `results/`.

## Generate challenge results of the runtime-optimized version

For our runtime-optimized version, additionally install [MMDeploy](https://github.com/open-mmlab/mmdeploy) and [TensorRT](https://developer.nvidia.com/tensorrt) following the instructions [here](https://github.com/open-mmlab/mmdeploy/blob/main/docs/en/get_started.md). If you do not have a GPU with CUDA Compute Capability 8.9, you have to convert the detection model to TensorRT on your specific GPU.

Please set `TMOT_DATASET_ROOT` and `GPU_ID` (optional) in `run_runtime_optimized_version.sh` accordingly.
The provided script `run_runtime_optimized_version.sh` performs the following tasks:
1. Write dataset root into detection config.
2. Generate detections per sequence and save them in `output_runtime_optimized/detections_trt/` as `seqXX.pkl`. The runtime is measured and saved in `output_runtime_optimized/detections_trt/runtime_trt_detector.txt`. 
3. Extract features with OSNet TRT model. Features will be saved per sequence together with the detections in `output_runtime_optimized/detections_trt_w_trt_osnet_features/`. The runtime will be measured and written in `output_runtime_optimized/detections_trt_w_trt_osnet_features_runtime.txt`.
4. Tracking is performed with the loaded detections and features. Tracking results are saved as `seqXX.txt` in `output_runtime_optimized/tracking/`. Tracking evaluation is performed and runtime is measured. The evaluation results can be found in `output_runtime_optimized/tracking/results.txt` and `output_runtime_optimized/tracking/pedestrian_summary.txt`. The measured runtime is saved in `output_runtime_optimized/tracking/tracker_runtime.txt`. The config file of the tracker is saved as `output_runtime_optimized/tracking/tracking.yaml`.

The generated files in `output_runtime_optimized/` should match the ones in `results_runtime_optimized/`.

## Reproduce other tracking results from the paper

Please have a look into `track.py` for a detailled description of all tracking parameters. In the follwining, commands for reproducing examplary experimental results from the paper are given.

#### Performance without NSA Kalman filter (Table 9, line 3)
```
python track.py ${DETECTIONS_DIR} ${OUTPUT_DIR} --use_nsa False
```

#### Performance without second association stage (Table 9, line 5)
```
python track.py ${DETECTIONS_DIR} ${OUTPUT_DIR} --use_second_stage False
```

#### Performance without appearance information (Table 9, line 2)

```
python track.py ${DETECTIONS_DIR} ${OUTPUT_DIR} --s1_metrics iou --s1_weights 1 --s1_dist_thresh 0.9 --s2_metrics iou --s2_weights 1 --s2_dist_thresh 0.5 --nsa_scale_factor 0.1 --cw_scale_factor 2.0
```
As you can see in the last example, if very influencial components such as the association measure is altered, other related parameters have to be tuned as well to achieve the best results.


## Citation

```
@InProceedings{Stadler_2025_CVPR,
    author    = {Stadler, Daniel and Specker, Andreas},
    title     = {A Strong Baseline for Multi-Person Tracking in Thermal Infrared Imagery},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2025},
}
```

## Acknowledgement

This repository contains code from [DeepSORT](https://github.com/nwojke/deep_sort), [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw), [MMYolo](https://github.com/open-mmlab/mmyolo/tree/main), [Torchreid](https://github.com/KaiyangZhou/deep-person-reid), [SOLIDER](https://github.com/tinyvision/SOLIDER-REID), and [TrackEval](https://github.com/JonathonLuiten/TrackEval). Many thanks for the excellent works. 