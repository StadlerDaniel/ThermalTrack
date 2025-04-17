import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import os
import time
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image

import sys
root_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(root_dir, 'SOLIDER-REID'))

from model import make_model
from config import cfg


class SOLIDERExtractor(nn.Module):
    @torch.no_grad()
    def __init__(self, config_file, device=None, weights_path=None):
        super().__init__()
        
        if config_file != "":
            cfg.merge_from_file(config_file)
        
        cfg.freeze()
        self.cfg = cfg
        
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.model = make_model(self.cfg, num_class=1, camera_num=0, view_num=0, semantic_weight=self.cfg.MODEL.SEMANTIC_WEIGHT)
        if weights_path is not None:
            self.model.load_param(weights_path)
        else:
            self.model.load_param(self.cfg.TEST.WEIGHT)
        self.model.cuda()
        self.model.eval()
        
        self.preproc = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        ])
    
    def preprocessing(self, xyxys, img):  # img: torch.Tensor (height, width, channels)
        crops = []
        for xyxy in xyxys:
            tmp_crop = self.preproc(
                img.crop((max(0, int(xyxy[0])), max(0, int(xyxy[1])), int(xyxy[2]), int(xyxy[3])))
            )
            crops.append(tmp_crop)
        crops = torch.stack(crops).to(self.device).type_as(next(self.model.parameters()))
        crops = crops.cuda()
        return crops
    
    @torch.no_grad()
    def forward(self, x, frame=None):
        if frame is not None:
            img = frame
            x = self.preprocessing(x, img)
        x = self.model(x)
        x = self.postprocessing(x)
        return x
    
    @staticmethod
    def postprocessing(feats):
        feats = feats[0]
        feats[torch.isinf(feats)] = 1.0
        feats = F.normalize(feats)
        return feats.cpu().data.numpy()


dets_dir = sys.argv[1]
sequences_dir = sys.argv[2]
config_path = sys.argv[3]
weights_path = sys.argv[4]
runtime_out_file = sys.argv[5]

feature_dim = 768

out_name = '_w_features'

out_dir = os.path.join(os.path.dirname(dets_dir), os.path.basename(dets_dir) + out_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

model = SOLIDERExtractor(config_path, weights_path=weights_path)

sequences = os.listdir(sequences_dir)

runtimes = []

for i, seq in enumerate(sequences):
    
    filename = seq + '.pkl'
    dets_path = os.path.join(dets_dir, filename)
    
    with open(dets_path, 'rb') as f:
        dets = pickle.load(f)
    
    seq_dir = os.path.join(sequences_dir, seq, 'thermal')
    images = sorted(os.listdir(seq_dir))
    
    assert len(images) == len(dets)
    
    # warmup
    if i == 0:
        print('Performing warmup ...')
        img = Image.open(os.path.join(seq_dir, images[0]))
        xyxys = dets[0][:, :4]
        for j in range(100):
            model(xyxys, img)
    
    dets_w_feats_seq = []
    
    for i in tqdm(range(len(images))):
        img = Image.open(os.path.join(seq_dir, images[i]))
        
        xyxys = dets[i][:, :4]
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        if xyxys.shape[0]:
            feats = model(xyxys, img)
        else:
            feats = np.empty((0, feature_dim))
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        runtimes.append(end_time - start_time)
        dets_w_feats = np.concatenate([dets[i], feats], axis=1)
        dets_w_feats_seq.append(dets_w_feats)
    
    out_path = os.path.join(out_dir, seq + '.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(dets_w_feats_seq, f)

median_runtime = np.median(runtimes) * 1000

with open(runtime_out_file, 'w') as f:
    f.write(f'Median runtime per image in ms:\t{median_runtime:.1f}\n')
