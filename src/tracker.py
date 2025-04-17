import numpy as np
import os
import yaml
import pickle
import cv2
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanFilter
from .misc import cosine_distance, box_diou, box_iou, write_results


def track(config):
    output_dir = config['output_dir']
    detections_dir = config['detections_dir']
    
    # save config
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, os.path.basename(output_dir) + '.yml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # initialize tracker
    tracker = Tracker(config)
    
    runtimes = []
    videos = ['seq2', 'seq17', 'seq22', 'seq47', 'seq54', 'seq66']
    
    for i, video in enumerate(videos):
        tracker.reset()
        
        # load detections with features
        with open(os.path.join(detections_dir, video + '.pkl'), 'rb') as f:
            detections = pickle.load(f)
        
        # tracking
        for i, dets in enumerate(tqdm(detections)):
            start_time = time.perf_counter()
            tracker.step(dets)
            end_time = time.perf_counter()
            runtimes.append(end_time - start_time)
        
        # save results
        results = tracker.get_results()
        
        # filter short tracks
        filter_min_len = config['filter_min_len']
        if filter_min_len:
            tids = results.keys()
            for tid in list(tids):
                if len(results[tid]) < filter_min_len:
                    del results[tid]
        
        # save results and runtime
        write_results(results, output_dir, video)
        median_runtime = np.median(runtimes) * 1000
        with open(os.path.join(output_dir, 'tracker_runtime.txt'), 'w') as f:
            f.write(f'Median runtime per image in ms:\t{median_runtime:.2f}\n')


class Tracker:
    def __init__(self, tracker_cfg):
        self.tracks = []
        self.results = {}
        self.frame = 1
        self.track_num = 0
        
        # inactive patience
        self.inactive_patience = tracker_cfg['inactive_patience']
        
        # kalman filter
        self.use_nsa = tracker_cfg['use_nsa']
        self.nsa_use_square = tracker_cfg['nsa_use_square']
        self.nsa_scale_factor = tracker_cfg['nsa_scale_factor']
        self.use_cw = tracker_cfg['use_cw']
        self.cw_score_thresh = tracker_cfg['cw_score_thresh']
        self.cw_scale_factor = tracker_cfg['cw_scale_factor']
        
        # track initialization
        self.init_min_score = tracker_cfg['init_min_score']
        self.n_dets_for_activation = tracker_cfg['n_dets_for_activation']
        
        # appearance
        self.first_feat_index = 5 if tracker_cfg['use_reid'] else None
        self.ema_alpha = tracker_cfg['ema_alpha']
        
        # association
        self.matching_stage_ids = tracker_cfg['matching_stages']
        self.matching_stage_cfgs = dict(zip(self.matching_stage_ids, [tracker_cfg['matching_stage_{}'.format(i)] for i in self.matching_stage_ids]))
    
    def reset(self):
        self.tracks = []
        self.results = {}
        self.frame = 1
        self.track_num = 0
    
    def add(self, new_dets):
        num_new = new_dets.shape[0]
        if self.first_feat_index is None:
            features = None
            for i in range(num_new):
                self.tracks.append(Track(new_dets[i, :4], new_dets[i, 4], features, self.track_num + 1 + i, frame=self.frame, first_feat_index=self.first_feat_index))
        else:
            features = new_dets[:, self.first_feat_index:]
            for i in range(num_new):
                self.tracks.append(Track(new_dets[i, :4], new_dets[i, 4], features[i, :], self.track_num + 1 + i, frame=self.frame, first_feat_index=self.first_feat_index))
        self.track_num += num_new
    
    @staticmethod
    def get_track_boxes(tracks):
        return np.stack([t.box for t in tracks], axis=0)
    
    @staticmethod
    def get_track_features(tracks):
        return np.stack([t.features for t in tracks], axis=0)
    
    def calc_dist_mat(self, tracks, dets, match_cfg):
        metric_dists = []
        
        for metric, weight in zip(match_cfg['metrics'], match_cfg['weights']):
            if metric == 'iou':
                det_boxes = dets[:, :4]
                track_boxes = self.get_track_boxes(tracks)
                iou = box_iou(track_boxes, det_boxes)
                dist = 1 - iou
            elif metric == 'diou':
                det_boxes = dets[:, :4]
                track_boxes = self.get_track_boxes(tracks)
                diou = box_diou(track_boxes, det_boxes)
                dist = 1 - diou
            elif metric == 'app_cos':
                track_features = self.get_track_features(tracks)
                det_features = dets[:, self.first_feat_index:]
                dist = cosine_distance(track_features, det_features, data_is_normalized=False)
            elif metric == 'app_l2':
                dist = np.zeros((len(tracks), dets.shape[0]))
                track_features = self.get_track_features(tracks)
                det_features = dets[:, self.first_feat_index:]
                for i, t_features in enumerate(track_features):
                    delta = t_features[None, :] - det_features
                    dists = np.linalg.norm(delta, axis=1)
                    dist[i, :] = dists
            elif metric == 'hist':
                dist = np.zeros((len(tracks), dets.shape[0]))
                track_features = self.get_track_features(tracks)
                det_features = dets[:, self.first_feat_index:]
                for i in range(track_features.shape[0]):
                    for j in range(det_features.shape[0]):
                        dist[i, j] = 1 - ((cv2.compareHist(track_features[i], det_features[j], cv2.HISTCMP_CORREL) + 1) / 2)
            
            metric_dists.append(weight * dist)
        
        dist_mat = sum(metric_dists)            
        return dist_mat
    
    def match(self, dist_mat, tracks, dets, dist_thresh):
        row, col = linear_sum_assignment(dist_mat)
        
        assigned_track_ids, assigned_det_idx = [], []
        for r, c in zip(row, col):
            if dist_mat[r, c] > dist_thresh:
                continue
            assigned_track_ids.append(tracks[r].id)
            assigned_det_idx.append(c)
        
        return assigned_track_ids, assigned_det_idx  
    
    def matching(self, dets):
        track_pool= self.tracks
        
        for match_stage in self.matching_stage_ids:
            match_cfg = self.matching_stage_cfgs[match_stage]
            
            stage_tracks = [t for t in track_pool if t.state in match_cfg['track_types']]
            
            min_score = match_cfg.get('min_score', 0.0)
            max_score = match_cfg.get('max_score', 1.0)
            is_stage_det = np.logical_and(max_score >= dets[:, 4], dets[:, 4] >= min_score)
            stage_dets = dets[is_stage_det]
            remaining_dets = dets[~is_stage_det]
            
            if len(stage_tracks) and stage_dets.shape[0]:
                dist_mat = self.calc_dist_mat(stage_tracks, stage_dets, match_cfg)
                ass_track_ids, ass_det_idx = self.match(dist_mat, stage_tracks, stage_dets, match_cfg['dist_thresh'])
                
                ass_tracks = [t for t in stage_tracks if t.id in ass_track_ids]
                ass_dets = stage_dets[ass_det_idx]
                if ass_dets.shape[0]:
                    Track.multi_update(ass_tracks, ass_dets, self.frame, self.use_nsa, self.nsa_use_square, self.nsa_scale_factor,
                                       self.use_cw, self.cw_score_thresh, self.cw_scale_factor, self.ema_alpha)
                
                track_pool = [t for t in track_pool if t.id not in ass_track_ids]
                dets = stage_dets[[i for i in range(len(stage_dets)) if i not in ass_det_idx]]
                dets = np.concatenate([dets, remaining_dets])
        
        # filter for initialization
        high_conf_dets = dets[dets[:, 4] >= self.init_min_score]
        
        return track_pool, high_conf_dets
    
    def association(self, dets):
        unassigned_tracks, unassigned_dets = self.matching(dets)
        
        # handle unassigned tracks
        remove = []
        for t in unassigned_tracks:
            if t.state == 'tentative':
                remove.append(t.id)
            elif t.state == 'active':
                t.state = 'inactive'
            elif t.state == 'inactive':
                if self.frame - t.frame_last_det > self.inactive_patience:
                    remove.append(t.id)
        self.tracks = [t for t in self.tracks if t.id not in remove]
        
        # start tentative tracks
        if unassigned_dets.shape[0]:
            self.add(unassigned_dets)
        
        # activate tenative tracks
        for t in [t for t in self.tracks if t.state == 'tentative']:
            if t.ndets == self.n_dets_for_activation:
                t.state = 'active'
    
    def step(self, dets):
        # kalman filter prediction
        if len(self.tracks):
            Track.multi_predict(self.tracks)
            means = np.zeros((len(self.tracks), 4))
            for i, t in enumerate(self.tracks):
                means[i] = t.mean[:4].copy()
            boxes = Track.xyah_to_boxes(means)
            for i in range(len(self.tracks)):
                self.tracks[i].box = boxes[i]
        
        # association of detections to tracks / start of new tracks
        self.association(dets)
        
        # immediate initialization in first frame (has no effect for self.n_dets_for_activation = 1)
        if self.frame == 1:
            for t in self.tracks:
                t.state = 'active'
        
        # save active track boxes
        self.save_current_results()
        
        self.frame += 1
    
    def save_current_results(self):
        for t in self.tracks:
            if t.state == 'active':
                if t.id not in self.results.keys():
                    self.results[t.id] = {}
                result = np.zeros(5)
                result[:4] = t.box
                result[4] = t.score
                self.results[t.id][self.frame] = result
    
    def get_results(self):
        return self.results


class Track(object):
    shared_kalman = KalmanFilter()
    
    def __init__(self, box, score, features, track_id, frame, first_feat_index):
        self.box = box
        self.score = score
        self.id = track_id
        self.frame_last_det = frame
        self.state = 'tentative'
        self.ndets = 1
        self.features = features
        self.first_feat_index = first_feat_index
        self.mean, self.covariance = Track.shared_kalman.initiate(Track.box_to_xyah(box))
    
    @staticmethod
    def multi_update(tracks, dets, frame, use_nsa=True, nsa_use_square=True, nsa_scale_factor=1.0, use_cw=False, cw_score_thresh=0.0, cw_scale_factor=1.0, ema_alpha=0.9):
        measurements = Track.boxes_to_xyah(dets[:, :4])
        scores = []
        
        for i in range(len(tracks)):
            t = tracks[i]
            det = dets[i]
            if t.state == 'inactive':
                t.state = 'active'
            if t.features is not None:
                det_features = det[t.first_feat_index:]
                track_features = t.features
                new_features = track_features * ema_alpha + det_features * (1 - ema_alpha)
                t.features = new_features / np.linalg.norm(new_features)
            t.score = det[4]
            t.frame_last_det = frame
            t.ndets += 1
            scores.append(det[4])
        
        multi_mean = np.asarray([t.mean.copy() for t in tracks])
        multi_covariance = np.asarray([t.covariance for t in tracks])
        scores = np.asarray(scores)
        multi_mean, multi_covariance = Track.shared_kalman.multi_update(multi_mean, multi_covariance, measurements, confidence=scores,
                                                                        use_nsa=use_nsa, nsa_use_square=nsa_use_square, nsa_scale_factor=nsa_scale_factor,
                                                                        use_cw=use_cw, cw_score_thresh=cw_score_thresh, cw_scale_factor=cw_scale_factor)
        track_means = np.zeros((len(tracks), 4))
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            tracks[i].mean = mean
            tracks[i].covariance = cov
            track_means[i] = mean[:4]
        boxes = Track.xyah_to_boxes(track_means)
        for i in range(len(tracks)):
            tracks[i].box = boxes[i]
    
    @staticmethod
    def multi_predict(tracks):
        if len(tracks) > 0:
            multi_mean = np.asarray([t.mean.copy() for t in tracks])
            multi_covariance = np.asarray([t.covariance for t in tracks])
            multi_mean, multi_covariance = Track.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                tracks[i].mean = mean
                tracks[i].covariance = cov
    
    @staticmethod
    def box_to_xyah(box):
        x, y, a, h = (box[0]+box[2])/2, (box[1]+box[3])/2, (box[2]-box[0])/(box[3]-box[1]), (box[3]-box[1])
        xyah = np.asarray([x, y, a, h])
        return xyah
    
    @staticmethod
    def boxes_to_xyah(boxes):
        xyah = np.zeros(boxes.shape)
        x = (boxes[:, 0] + boxes[:, 2]) / 2
        y = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        a = w / h
        xyah[:, 0] = x
        xyah[:, 1] = y
        xyah[:, 2] = a
        xyah[:, 3] = h
        return xyah
    
    @staticmethod
    def xyah_to_boxes(xyah):
        boxes = np.zeros(xyah.shape)
        x, y, a, h = xyah[:, 0], xyah[:, 1], xyah[:, 2], xyah[:, 3]
        w = a * h
        x1, x2, y1, y2 = x - w/2,  x + w/2, y - h/2,  y + h/2
        boxes[:, 0] = x1
        boxes[:, 1] = y1
        boxes[:, 2] = x2
        boxes[:, 3] = y2
        return boxes
