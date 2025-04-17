import argparse

from src.tracker import track
from src.misc import evaluate, get_config, str2bool, list_of_strings, list_of_floats


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Track end evaluate on TMOT val split.')
    # general settings
    parser.add_argument('detections_dir', help='directory of detection pickle files')
    parser.add_argument('output_dir', help='directory where tracking (and evaluation) results are saved')
    parser.add_argument('--eval', default=True, type=str2bool, help='evaluate HOTA, MOTA, and IDF1')
    parser.add_argument('--filter_min_len', default=5, type=int, help='remove short tracks')
    # inactive patience
    parser.add_argument('--inactive_patience', type=int, default=40, help='number of frames a track remains inactive before termination')
    # kalman filter
    parser.add_argument('--use_nsa', default=True, type=str2bool, help='whether to use noise scale adaptive (NSA) Kalman filter')
    parser.add_argument('--nsa_use_square', default=True, type=str2bool, help='whether to use squared scaling in NSA equation')
    parser.add_argument('--nsa_scale_factor', default=0.15, type=float, help='scale factor in NSA equation')
    parser.add_argument('--use_cw', default=True, type=str2bool, help='whether to use confidence-weighted Kalman update')
    parser.add_argument('--cw_score_thresh', default=0.4, type=float, help='score thresh in confidence-weighted Kalman update')
    parser.add_argument('--cw_scale_factor', default=2.5, type=float, help='scale factor in confidence-weighted Kalman update')
    # track initialization
    parser.add_argument('--init_min_score', type=float, default=0.7, help='minimum detection confidence to initialize a track')
    parser.add_argument('--n_dets_for_activation', type=int, default=1, help='number of required consecutive detections to activate a tentative track')
    # appearance
    parser.add_argument('--use_reid', default=True, type=str2bool, help='whether to use appearance features')
    parser.add_argument('--ema_alpha', type=float, default=0.9, help='weighting parameter in EMA feature update')
    # 1st association stage
    parser.add_argument('--s1_min_score', default=0.6, type=float, help='minimum detection confidence in stage 1')
    parser.add_argument('--s1_metrics', default=['diou', 'app_l2'], type=list_of_strings, help='matching metrics in stage 1')
    parser.add_argument('--s1_weights', default=[1, 20], type=list_of_floats, help='matching weights in stage 1')
    parser.add_argument('--s1_dist_thresh', default=3.2, type=float, help='distance threshold in stage 1')
    parser.add_argument('--s1_track_types', default=['active', 'inactive', 'tentative'], nargs='+', help='track types considered in stage 1')
    # 2nd association stage
    parser.add_argument('--use_second_stage', default=True, type=str2bool, help='whether to use a second association stage')
    parser.add_argument('--s2_min_score', default=0.2, type=float, help='minimum detection confidence in stage 2')
    parser.add_argument('--s2_max_score', default=0.6, type=float, help='maximum detection confidence in stage 2')
    parser.add_argument('--s2_metrics', default=['diou', 'app_l2'], type=list_of_strings, help='matching metrics in stage 2')
    parser.add_argument('--s2_weights', default=[1, 20], type=list_of_floats, help='matching weights in stage 2')
    parser.add_argument('--s2_dist_thresh', default=2.4, nargs='+', type=float, help='distance threshold in stage 2')
    parser.add_argument('--s2_track_types', default=['active'], nargs='+', help='track types considered in stage 2')
    
    args = parser.parse_args()
    args.output_dir = args.output_dir.rstrip('/')
    config = get_config(args)
    track(config)
    
    if args.eval:
        evaluate(config)
