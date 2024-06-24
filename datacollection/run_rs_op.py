import argparse

from datacollection.utils import extract_pose_from_heatmaps
from datacollection.utils import save_heatmaps

from openpose.native.python.utils import str2bool

from rs_py.rs_run_devices import get_rs_parser

from openpose.native.python.args import get_parser as get_op_parser


def get_rs_args():
    args, _ = get_rs_parser().parse_known_args()
    # args.rs_steps = 60
    # args.rs_fps = 15
    # args.rs_image_width = 848
    # args.rs_image_height = 480
    # args.rs_save_data = True
    # args.rs_save_path = 'tmp/data'
    # args.rs_verbose = False
    print("========================================")
    print(">>>>> rs_args <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


def get_op_args():
    args, _ = get_op_parser().parse_known_args()
    # # args.op_net_resolution = "-1x368"
    # # args.op_skel_thres = 0.2
    # # args.op_max_true_body = 8
    # args.op_image_width = 848
    # args.op_image_height = 480
    # # # For 3d skel extraction
    # args.op_patch_offset = 5
    # # args.op_extract_3d_skel = False
    # # args.op_save_3d_skel = False
    print("========================================")
    print(">>>>> op_args <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


if __name__ == "__main__":

    p = argparse.ArgumentParser(description='Run OPENPOSE on RS images')
    p.add_argument('--extract-pose',
                   type=str,
                   default="",
                   help='path to saved rs data')
    p.add_argument('--display-pose',
                   type=str2bool,
                   default=False,
                   help='')
    p.add_argument('--save-heatmaps',
                   type=str2bool,
                   default=False,
                   help='')
    p.add_argument('--face',
                   type=str2bool,
                   default=True,
                   help='enable face detection')

    args, _ = p.parse_known_args()

    rs_args = get_rs_args()
    op_args = get_op_args()

    if len(args.extract_pose) != 0:
        extract_pose_from_heatmaps(args.extract_pose, op_args,
                                   args.display_pose, args.face)
    elif args.save_heatmaps:
        save_heatmaps(rs_args, op_args)
    else:
        raise ValueError("No arg given...")
