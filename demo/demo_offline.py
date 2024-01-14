import cv2
import json
import numpy as np
import os
import time
import sys
import traceback
import matplotlib.pyplot as plt

from rs_py.utility import get_filepaths_with_timestamps
from rs_py.utility import read_color_file
from rs_py.utility import read_depth_file
from rs_py.rs_run_devices import get_rs_parser
from rs_py.rs_run_devices import printout

from openpose.native.python.inference_rs import get_parser as get_op_parser
from openpose.native.python.utils import Timer

from utils.parser import get_parser as get_ar_parser
from utils.parser import load_parser_args_from_config
from utils.utils import init_seed

from demo import App
from demo import Config

FPS = 15
CAMERA_W = 848
CAMERA_H = 480

HORITONTAL_BAR = np.ones((20, CAMERA_W, 3), dtype=np.uint8)*255
VERTICAL_BAR = np.ones((CAMERA_W, 20, 3), dtype=np.uint8)*255

FIG, POSE, EDGE = None, None, None
FIG = plt.figure(figsize=(16, 8))


OUTPUT_VIDEO = None
OUTPUT_VIDEO = cv2.VideoWriter(
    '/data/tmp/demo_v2_test.avi',
    cv2.VideoWriter_fourcc(*'DIVX'),
    15,
    (1480, 848)
)

config = Config()
config.BUFFER = 0
config.delay_switch = 0
config.delay_counter = 0
config.MOVING_AVG = 5  # 5
config.MAY_YS_DELAY = 3
config.MAY_YS_COUNT = 8
config.MAY_YL_DELAY = 6
config.MAY_YL_COUNT = 11
config.MOTION_DELAY = 5
config.MOTION_COUNT = 10  # 10
config.ACTION_COUNT = 5
config.MOVE_THRES = 0.004  # 0.015
config.RULE_THRES_STAND = 0.05  # 0.05
config.RULE_THRES_STAND_LONG = 0.10  # 0.10
config.RULE_THRES_SIT = -0.05  # -0.05
config.RULE_THRES_SIT_LONG = -0.10  # -0.10
config.RULE_THRES_FALL = -0.1  # -0.1
config.RULE_THRES_FALL_HEIGHT = 1.30  # 1.25
DATA_PATH = '/data/realsense_230523'
TRIAL_IDX = 0


class AppExt(App):
    def __init__(self) -> None:
        super().__init__()
        pass

    def read_offline_data(self, data_path, trial_idx):
        color_dict, depth_dict, _, trial_list = \
            get_filepaths_with_timestamps(data_path)
        trial_id = trial_list[trial_idx]
        device_sn = list(color_dict[trial_id].keys())[0]
        color_filepaths = [v[0] for _, v in
                           color_dict[trial_id][device_sn].items()]
        depth_filepaths = [v[0] for _, v in
                           depth_dict[trial_id][device_sn].items()]
        color_calibs = [v[1] for _, v in
                        color_dict[trial_id][device_sn].items()]
        depth_calibs = [v[1] for _, v in
                        depth_dict[trial_id][device_sn].items()]
        # os.makedirs('/data/tmp/depth', exist_ok=True)
        # for depth_filepath in depth_filepaths:
        #     output_filepath = depth_filepath.split('/')[-1].replace('npy',
        #                                                             'bin')
        #     output_filepath = os.path.join('/data/tmp/depth', output_filepath)
        #     depth_image = read_depth_file(depth_filepath)
        #     depth_image = depth_image[-848*480:]
        # #     depth_image.astype(np.uint16).tofile(output_filepath)
        return (device_sn, color_filepaths, depth_filepaths,
                color_calibs, depth_calibs)


def get_args_rs():
    args, _ = get_rs_parser().parse_known_args()
    args.rs_steps = 1000
    args.rs_fps = FPS
    args.rs_image_width = CAMERA_W
    args.rs_image_height = CAMERA_H
    args.rs_ir_emitter_power = 290
    args.rs_display_frame = 0
    args.rs_save_with_key = False
    args.rs_save_data = False
    args.rs_save_path = ''
    args.rs_postprocess = True  # max fps ~25 for 1280x720
    args.rs_vertical = True
    print("========================================")
    print(">>>>> args_rs <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


def get_args_op():
    args, _ = get_op_parser().parse_known_args()
    args.op_model_folder = "/usr/local/src/openpose/models/"
    args.op_model_pose = "BODY_25"
    args.op_net_resolution = "-1x368"
    args.op_skel_thres = 0.1
    args.op_max_true_body = 4
    # args.op_heatmaps_add_parts = True
    # args.op_heatmaps_add_bkg = True
    # args.op_heatmaps_add_PAFs = True
    args.op_save_skel_folder = ""
    args.op_save_skel = False
    args.op_save_skel_image = False
    # For skel extraction/tracking in inference.py
    args.op_input_color_image = ""
    args.op_image_width = CAMERA_W
    args.op_image_height = CAMERA_H
    # # For 3d skel extraction
    args.op_patch_offset = 3
    # # For 3d skel extraction
    # args.op_extract_3d_skel = False
    # args.op_save_3d_skel = False
    args.op_display = 1.0
    # args.op_display_depth = 0  # Not used
    # For skel extraction/tracking in inference_rs.py
    args.op_rs_dir = "data/mot17"
    args.op_rs_delete_image = False
    args.op_save_track_image = False
    args.op_proc = "sp"
    # args.op_track_deepsort = True
    args.op_track_bytetrack = True
    # args.op_track_ocsort = True
    # args.op_track_strongsort = True
    args.op_track_buffer = FPS
    args.bytetracker_trackthresh = 0.25
    # args.bytetracker_trackbuffer = 30  # overwritten by op_track_buffer
    args.bytetracker_matchthresh = 0.98
    args.bytetracker_mot20 = False
    args.bytetracker_fps = FPS
    print("========================================")
    print(">>>>> args_op <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


def get_args_ar():
    init_seed(1)
    # label_mapping_file = "/data/07_AAGCN/model/ntu_15j/index_to_name.json"
    # label_mapping_file = "/data/07_AAGCN/model/ntu_15j_9l/index_to_name.json"
    # label_mapping_file = "/data/07_AAGCN/model/ntu_15j_5l/index_to_name.json"
    label_mapping_file = "/data/07_AAGCN/model/ntu_15j_4l/index_to_name.json"
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/weight/SGN-116-68208.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/weight/SGN-110-10670.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_5l_ntu_result/xview/aagcn_preprocess_sgn_model/230511130001/weight/SGN-120-8160.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_4l_ntu_result/xview/aagcn_preprocess_sgn_model/230512123001/weight/SGN-66-3168.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_4l_ntu_result/xsub/aagcn_preprocess_sgn_model/230512123001/weight/SGN-112-5824.pt"  # noqa
    weights = "/data/07_AAGCN/data/openpose_b25_j11_4l_ntu_result/xview/aagcn_preprocess_sgn_model/230517123001/weight/SGN-94-4512.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j11_4l_ntu_result/xsub/aagcn_preprocess_sgn_model/230517123001/weight/SGN-82-4264.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_5l_ntu_result/xview/aagcn_joint/230511130001/weight/Model-40-2720.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_joint/230414100001/weight/Model-50-29400.pt"  # noqa
    # weights = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_joint/230414100001/weight/Model-50-4850.pt"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_preprocess_sgn_model/230414100001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_5l_ntu_result/xview/aagcn_preprocess_sgn_model/230511130001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_4l_ntu_result/xview/aagcn_preprocess_sgn_model/230512123001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_4l_ntu_result/xsub/aagcn_preprocess_sgn_model/230512123001/config.yaml"  # noqa
    config = "/data/07_AAGCN/data/openpose_b25_j11_4l_ntu_result/xview/aagcn_preprocess_sgn_model/230517123001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j11_4l_ntu_result/xsub/aagcn_preprocess_sgn_model/230517123001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_5l_ntu_result/xview/aagcn_joint/230511130001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_ntu_result/xview/aagcn_joint/230414100001/config.yaml"  # noqa
    # config = "/data/07_AAGCN/data/openpose_b25_j15_9l_ntu_result/xview/aagcn_joint/230414100001/config.yaml"  # noqa
    parser = get_ar_parser()
    parser.set_defaults(**{'config': config})
    args = load_parser_args_from_config(parser)
    args.weights = weights
    args.max_frame = 45
    args.max_num_skeleton_true = 2
    args.max_num_skeleton = 4
    args.num_joint = 11
    args.gpu = True
    args.timing = False
    args.interval = 0
    args.moving_avg = 1
    args.aagcn_normalize = True
    args.sgn_preprocess = True
    args.multi_test = 5
    # args.data_path = data_path
    args.label_mapping_file = label_mapping_file
    # args.out_folder = "/data/07_AAGCN/data_tmp/delme"
    print("========================================")
    print(">>>>> args_ar <<<<<")
    print("========================================")
    for k, v in vars(args).items():
        print(f"{k} : {v}")
    print("========================================")
    return args


if __name__ == "__main__":

    # fp = np.load('sandbox/realsense_230516.npy')  # M, T, V, C
    # fp = fp.transpose(3, 1, 2, 0)[:, -50:, :, :]  # C, T, V, M

    # num_labels = 4
    # x, y, z = [], [], []
    # for i in range(num_labels):
    #     x += [fp[0].reshape(-1)]
    #     y += [fp[1].reshape(-1)]
    #     z += [fp[2].reshape(-1)]

    # FIG, axes = plt.subplots(nrows=2, ncols=num_labels//2)
    # axes = axes.flatten()
    # for i in range(num_labels):
    #     axes[i].hist(x[i][x[i] != 0.0], bins=100, alpha=0.5, label='x')
    #     axes[i].hist(y[i][y[i] != 0.0], bins=100, alpha=0.5, label='y')
    #     axes[i].hist(z[i][z[i] != 0.0], bins=100, alpha=0.5, label='z')
    #     axes[i].legend(loc='upper right')
    #     # axes[i].set_xlim([-2, 5])
    #     # axes[i].set_ylim([0, 500])
    # plt.tight_layout()

    # FIG, axes = plt.subplots(nrows=1, ncols=3)
    # axes = axes.flatten()
    # for i in range(num_labels):
    #     axes[0].hist(x[i][x[i] != 0.0], bins=100, alpha=0.5, label=str(i))
    #     axes[1].hist(y[i][y[i] != 0.0], bins=100, alpha=0.5, label=str(i))
    #     axes[2].hist(z[i][z[i] != 0.0], bins=100, alpha=0.5, label=str(i))
    # for i in range(3):
    #     axes[i].legend(loc='upper right')
    #     # axes[i].set_xlim([-2, 5])
    #     # axes[i].set_ylim([0, 500])
    # plt.tight_layout()
    # plt.show()

    # exit(1)

    app = AppExt()
    app.setup(config, get_args_rs, get_args_op, get_args_ar)

    # setup realsense ----------------------------------------------------------
    # read data
    device_sn, color_filepaths, depth_filepaths, color_calibs, depth_calibs = \
        app.read_offline_data(DATA_PATH, TRIAL_IDX)
    app.RS_INFO['device_sn'] = device_sn

    # MAIN LOOP ----------------------------------------------------------------
    ar_start_time = time.time()

    try:
        c = 0
        max_c = int(1e8)

        while True:

            skel_added = False

            if c < 15:
                fps = {'PE': [], 'TK': []}

            # if c < 220:
            #     c += 1
            #     continue

            # 1. grab image from realsense -------------------------------------
            with open(color_calibs[c]) as f:
                calib = json.load(f)
            h_c = calib['color']['height']
            w_c = calib['color']['width']
            h_d = calib['depth']['height']
            w_d = calib['depth']['width']
            app.RS_INFO['intr_mat'] = calib['depth']['intrinsic_mat']
            app.RS_INFO['depth_scale'] = calib['depth']['depth_scale']
            color_image = read_color_file(color_filepaths[c],
                                          calib['color']['format'])
            color_image = color_image.reshape((h_c, w_c, -1))
            depth_image = read_depth_file(depth_filepaths[c])
            try:
                depth_image = depth_image[-h_d*w_d:].reshape(h_d, w_d)
            except ValueError:
                depth_image = depth_image[-h_d*w_d//4:].reshape(h_d//2, w_d//2)
            if app.CF.args_rs.rs_postprocess:
                depth_image = cv2.resize(depth_image, (color_image.shape[1],
                                                       color_image.shape[0]))
            color_image[:20, :, :] = 0
            color_image[-20:, :, :] = 0
            color_image[depth_image > 2800] = 0
            color_image[depth_image < 300] = 0

            # 2. track without pose extraction ---------------------------------
            # 3. infer pose and track ------------------------------------------
            status, op_image, skel_added = app.infer_pose_and_track(
                color_image, depth_image)
            if not status:
                break

            # if config.args_rs.rs_vertical:
            #     cv2.imshow("openpose", np.rot90(status[1], -1).copy())
            # else:
            #     cv2.imshow("openpose", status[1])
            # cv2.waitKey(1)

            # 4. action recognition --------------------------------------------
            if time.time() - ar_start_time < int(app.CF.args_ar.interval):
                continue

            with Timer("AR", app.CF.enable_timer, False) as t:
                # if len(app.DS.skeletons) > 0:
                if skel_added:
                    (raw_kypts, raw_score, avg_skels,
                     motion, prediction) = app.infer_action(c)
                else:
                    print(f'{c:04}',
                          f"No skeletons... {len(app.DS.avg_skels)}",
                          app.ET.PE.pyop.pose_scores)

            app.CF.recog_time = t.duration
            if app.CF.recog_time is None:
                app.CF.recog_time = -1

            # Visualize results -----------
            # ------ visualize 3d plot
            # if skel_added:
            #     # M, 1, V, C -> C, 1, V, M
            #     avg_skels = avg_skels.transpose(3, 1, 2, 0)[:, :, :, 0:1]  # noqa
            #     avg_skels[1] *= -1.0
            #     alpha = raw_score[0, 0].tolist()
            #     num_joint = app.CF.args_ar.num_joint
            #     app.visualize_3dplot(avg_skels, alpha, num_joint)

            # ------ visualize image
            _kpt = None
            if skel_added:
                assert op_image is not None
                text_id, text, color = app.TP.text_parser(
                    app.DS.ids[0], prediction, motion[0, 0])
                if raw_score[0, 0, 0] > 0.0:
                    _kpt = raw_kypts[0, 0, 0]

            else:
                assert op_image is not None
                op_image = np.zeros_like(op_image)
                text_id, text, color = app.TP.text_parser(-1, -1, -1)

            print(text)
            image = app.visualize_output(
                color_image,
                _kpt,
                text,
                color,
                app.CF.args_rs.rs_vertical
            )
            if app.CF.args_rs.rs_vertical:
                op_image = np.rot90(op_image, -1).copy()
            history = app.AH.step(color, text, text_id)
            image = np.hstack(
                [image, VERTICAL_BAR,
                 op_image, VERTICAL_BAR,
                 history]
            )

            if OUTPUT_VIDEO is not None:
                OUTPUT_VIDEO.write(image)
            else:
                cv2.imshow(f"est_{app.RS_INFO['device_sn']}", image)
                key = cv2.waitKey(0)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    break

            # 8. printout ------------------------------------------------------
            # if c % app.CF.args_rs.rs_fps == 0:
            #     # printout(
            #     #     f"Step {c:12d} :: "
            #     #     f"{[i.get('color_timestamp', None) for i in app.RS.frames.values()]} :: "  # noqa
            #     #     f"{[i.get('depth_timestamp', None) for i in app.RS.frames.values()]}",  # noqa
            #     #     'i'
            #     # )
            #     # color_timestamp = list(app.RS.frames.values())[0].get('color_timestamp')  # noqa
            #     color_timestamp = c
            #     fps['PE'].append(1/app.CF.infer_time)
            #     fps['TK'].append(1/app.CF.track_time)
            #     printout(
            #         f"Image : {color_timestamp} | "
            #         f"#Skel filtered : {app.CF.filered_skel} | "
            #         f"#Tracks : {len(app.ET.TK.tracks)} | "
            #         f"RS time : {app.CF.rs_time:.3f} | "
            #         f"Prep time : {app.CF.prep_time:.3f} | "
            #         f"Pose time : {app.CF.infer_time:.3f} | "
            #         f"Track time : {app.CF.track_time:.3f} | "
            #         f"Recog time : {app.CF.recog_time:.3f} | "
            #         f"FPS PE : {sum(fps['PE'])/len(fps['PE']):.3f} | "
            #         f"FPS TK : {sum(fps['TK'])/len(fps['TK']):.3f}",
            #         'i'
            #     )

            # if not len(rsw.frames) > 0:
            #     printout(f"Empty...", 'w')
            #     continue

            if c % (app.CF.args_rs.rs_fps*60) == 0 and c > 0:
                app.DS.reset()

            c += 1

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        printout(f"{exc_type}, {fname}, {exc_tb.tb_lineno}", 'e')
        printout(f"Exception msg : {e}", 'e')
        traceback.print_tb(exc_tb)
        printout(f"Stopping RealSense devices...", 'i')
        # rsw.stop()

    finally:
        printout(f"Final RealSense devices...", 'i')
        # rsw.stop()

    # fp = np.concatenate(data_to_save, axis=1)  # m,t,v,c
    # np.save('./sandbox/realsense_230517.npy', fp)

    if OUTPUT_VIDEO is not None:
        OUTPUT_VIDEO.release()
    app.ET.break_process_loops()
    printout(f"Finished...", 'i')
