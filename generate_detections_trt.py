import numpy as np
import os
import torch
from tqdm import tqdm
import pickle
import time
import sys

from mmdeploy.utils import get_input_shape, load_config
from mmdeploy.apis.utils import build_task_processor


deploy_cfg = sys.argv[1]
model_cfg = sys.argv[2]
engine_file = sys.argv[3]
out_dir = sys.argv[4]
device = 'cuda:0'

backend_files = [engine_file]

runtime_out_file = os.path.join(out_dir, 'runtime_trt_detector.txt')
runtimes = []

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
task_processor = build_task_processor(model_cfg, deploy_cfg, device)
model = task_processor.build_backend_model(backend_files, task_processor.update_data_preprocessor)
input_shape = get_input_shape(deploy_cfg)

sequences_dir = '/net/vid-ssd1/storage/deeplearning/datasets/TP-MOT/tmot_dataset/images/val'
sequences = os.listdir(sequences_dir)

for seq in sequences:
    seq_dir = os.path.join(sequences_dir, seq, 'thermal')
    images = sorted(os.listdir(seq_dir))
    np.array(images)
    
    detections = []
    for image in tqdm(images):
        img_path = os.path.join(seq_dir, image)
        model_inputs, _ = task_processor.create_input(img_path, input_shape)
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            result = model.test_step(model_inputs)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        runtimes.append(end_time - start_time)
        bboxes = result[0].pred_instances.bboxes.cpu().numpy()
        scores = result[0].pred_instances.scores.cpu().numpy()
        dets = np.concatenate([bboxes, scores[:, None]], axis=1)
        detections.append(dets)
    
    with open(os.path.join(out_dir, seq + '.pkl'), 'wb') as f:
        pickle.dump(detections, f)

median_runtime = np.median(runtimes) * 1000
        
with open(runtime_out_file, 'w') as f:
    f.write(f'Median runtime per image in ms:\t{median_runtime:.1f}\n')
