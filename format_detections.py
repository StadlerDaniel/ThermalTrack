import pickle
import os
import numpy as np
import sys


dets_path = sys.argv[1]
sequences_dir = sys.argv[2]

min_score = 0.2

out_dir = os.path.join(os.path.dirname(dets_path), os.path.basename(dets_path).split('.pkl')[0])
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

seqs = os.listdir(sequences_dir)

with open(dets_path, 'rb') as f:
    all_dets = pickle.load(f)

seq_lengths = []
for seq in seqs:
    seq_dir = os.path.join(sequences_dir, seq, 'thermal')
    frames = len(os.listdir(seq_dir))
    
    seq_dets = all_dets[:frames]
    all_dets = all_dets[frames:]
    
    confident_dets = []
    for dets in seq_dets:
        formatted_dets = np.concatenate([dets['pred_instances']['bboxes'].numpy(),
                                         dets['pred_instances']['scores'].numpy()[:, None]], axis=1)
        confident_dets.append(formatted_dets[formatted_dets[:, 4] >= min_score])
    
    out_path = os.path.join(out_dir, seq + '.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(confident_dets, f)

assert len(all_dets) == 0
