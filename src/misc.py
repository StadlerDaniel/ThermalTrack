import os
import numpy as np
import argparse
import sys
import inspect

trackeval_dir = os.path.join(os.path.dirname(os.path.dirname(inspect.getfile(lambda: None))), 'TrackEval')
sys.path.append(trackeval_dir)
import trackeval


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def list_of_strings(arg):
    return arg.split(',')

def list_of_floats(arg):
    return [float(a) for a in arg.split(',')]

def get_config(args):
    config = dict(detections_dir = args.detections_dir,
                  output_dir = args.output_dir,
                  filter_min_len = args.filter_min_len,
                  inactive_patience = args.inactive_patience,
                  use_nsa = args.use_nsa,
                  nsa_use_square = args.nsa_use_square,
                  nsa_scale_factor = args.nsa_scale_factor,
                  use_cw = args.use_cw,
                  cw_score_thresh = args.cw_score_thresh,
                  cw_scale_factor = args.cw_scale_factor,
                  init_min_score = args.init_min_score,
                  n_dets_for_activation = args.n_dets_for_activation,
                  use_reid = args.use_reid,
                  ema_alpha = args.ema_alpha,
                  matching_stages = [1],
                  matching_stage_1 = dict(track_types = args.s1_track_types,
                                          min_score = args.s1_min_score,
                                          metrics = args.s1_metrics,
                                          weights = args.s1_weights,
                                          dist_thresh = args.s1_dist_thresh))
    if args.use_second_stage:
        config['matching_stages'] = [1, 2]
        config['matching_stage_2'] = dict(track_types = args.s2_track_types,
                                          min_score = args.s2_min_score,
                                          max_score = args.s2_max_score,
                                          metrics = args.s2_metrics,
                                          weights = args.s2_weights,
                                          dist_thresh = args.s2_dist_thresh)
    return config

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2, return_union=False):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]
    wh = np.clip(rb - lt, 0, np.inf)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
    union = area1[:, None] + area2[None, :] - inter # [N,M]
    iou = inter / union
    if return_union:
        return iou, union
    else:
        return iou

def box_diou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2, return_union=True)

    c1 = (boxes1[:, None, :2] + boxes1[:, None, 2:]) / 2
    c2 = (boxes2[:, :2] + boxes2[:, 2:]) / 2
    inner_diag = ((c1 - c2) ** 2).sum(axis=2)
    
    lt = np.minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = np.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    outer_diag = ((lt - rb) ** 2).sum(axis=2)
    
    diou = iou - inner_diag / outer_diag
    return diou

def cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)

def write_results(results, output_dir, video):
    save_format = '{:d},{:d},{:.02f},{:.02f},{:.02f},{:.02f},{:.02f},1,-1,-1\n'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    results_list = []
    for i, track in results.items():
        for frame, det in track.items():
            x1 = det[0]
            y1 = det[1]
            w = det[2] - det[0]
            h = det[3] - det[1]
            score = det[4]
            res = np.array([frame, i, x1, y1, w, h, score])
            results_list.append(res)
    if results_list:
        results = np.stack(results_list)
        results = results[np.lexsort((results[:, 1], results[:, 0]))]
        file = os.path.join(output_dir, video + '.txt')
        with open(file, 'w') as f:
            for res in results:
                frame_id, track_id, x1, y1, w, h, s = res
                line = save_format.format(int(frame_id), int(track_id), x1, y1, w, h, s)
                f.write(line)

def evaluate(config):
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config['DISPLAY_LESS_PROGRESS'] = False
    eval_config['OUTPUT_EMPTY_CLASSES'] = False
    eval_config['OUTPUT_DETAILED'] = False
    eval_config['PLOT_CURVES'] = False
    evaluator = trackeval.Evaluator(eval_config)
    
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config['SKIP_SPLIT_FOL'] = True
    dataset_config['TRACKER_SUB_FOLDER'] = ''
    dataset_config['TRACKERS_FOLDER'] = os.path.dirname(config['output_dir'])
    dataset_config['TRACKERS_TO_EVAL'] = [os.path.basename(config['output_dir'])]
    dataset_config['GT_FOLDER'] = os.path.join(trackeval_dir, 'data/gt/tmot/tmot-val/')
    dataset_config['SEQMAP_FILE'] = os.path.join(trackeval_dir, 'data/gt/tmot/seqmaps/tmot-val.txt')
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    
    metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator.evaluate(dataset_list, metrics_list)
