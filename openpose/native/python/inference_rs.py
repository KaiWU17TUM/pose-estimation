import argparse
import os
from tqdm import tqdm

from openpose.native.python.args import get_parser
from openpose.native.python.inference import ExtractSkeletonAndTrack
from openpose.native.python.skeleton import OpenPosePoseExtractor
from openpose.native.python.utils import dict_check
from openpose.native.python.utils import Error
from openpose.native.python.utils_rs import get_rs_sensor_dir
from openpose.native.python.utils_rs import read_calib_file
from openpose.native.python.utils_rs import read_color_file

from tracking.track import Tracker


def rs_extract_skeletons_and_track_offline_mp(args: argparse.Namespace):
    """Runs openpose inference and tracking on realsense camera in offline mode.

    Reads realsense image files under the `base_path` arg and extracts pose
    from the images using openpose.

    Args:
        args (argparse.Namespace): CLI arguments
    """

    assert os.path.isdir(args.op_rs_dir), f'{args.op_rs_dir} does not exist...'

    base_path = args.op_rs_dir
    dev_trial_color_dir = get_rs_sensor_dir(base_path, 'color')
    dev_list = list(dev_trial_color_dir.keys())

    empty_dict = {i: Error() for i in dev_list}
    end_loop = False

    # Delay = predict and no update in tracker
    delay_switch = 5
    delay_counter = 0

    # For cv.imshow
    display_speed = 1

    # Runtime logging
    enable_timer = True
    runtime = {'PE': [], 'TK': []}

    # Setup extract and track classes ------------------------------------------
    PoseExtractor = OpenPosePoseExtractor(args)
    PoseTracker = Tracker(args, 30//(delay_switch+1))
    EST = ExtractSkeletonAndTrack(
        args, PoseExtractor, PoseTracker, enable_timer)
    EST.start()

    # 1. If no error -----------------------------------------------------------
    while not dict_check(empty_dict) and not end_loop:

        filepath_dict = {i: [] for i in dev_list}

        # 2. loop through devices ----------------------------------------------
        for dev, trial_color_dir in dev_trial_color_dir.items():

            # 3. loop through trials -------------------------------------------
            for trial, color_dir in trial_color_dir.items():

                color_filepaths = [os.path.join(color_dir, i)
                                   for i in sorted(os.listdir(color_dir))]

                if len(color_filepaths) == 0:
                    print(f"[INFO] : {color_dir} is empty...")
                    empty_dict[dev].counter += 1
                    if empty_dict[dev].counter > 300:
                        print("[INFO] Retried 300 times and no new files...")
                        empty_dict[dev].state = True
                    continue

                filepath_dict[dev] += color_filepaths

        # 4. loop through devices for offline inference ------------------------
        for dev, color_filepaths in filepath_dict.items():

            _c = 0

            tqdm_bar = tqdm(color_filepaths, dynamic_ncols=True)
            data_len = len(color_filepaths)
            break_loop = False

            calib_file = os.path.dirname(os.path.dirname(color_filepaths[0]))
            calib_file = calib_file + "/calib/calib.csv"
            if os.path.exists(calib_file):
                intr_mat = read_calib_file(calib_file)
            else:
                intr_mat = None

            # 5. loop through filepaths of color image -------------------------
            for idx, color_filepath in enumerate(tqdm_bar):

                if idx + 1 == data_len:
                    break_loop = True

                if idx == 15:
                    runtime = {'PE': [], 'TK': []}

                # if _c < 590:
                #     _c += 1
                #     continue

                depth_filepath = color_filepath.replace(
                    color_filepath.split('/')[-2], 'depth')
                skel_filepath, skel_prefix = os.path.split(color_filepath)
                skel_filepath = os.path.join(
                    os.path.split(skel_filepath)[0], 'skeleton')
                skel_prefix = os.path.splitext(skel_prefix)[0]

                # 6. track without pose extraction -----------------------------
                if delay_counter > 0:
                    delay_counter -= 1
                    _image = read_color_file(color_filepath)
                    EST.TK.no_measurement_predict_and_update()
                    EST.PE.display(win_name=dev,
                                   speed=display_speed,
                                   scale=args.op_display,
                                   image=None,
                                   bounding_box=True,
                                   tracks=EST.TK.tracks)

                else:
                    delay_counter = delay_switch

                    # 7. infer pose and track ----------------------------------
                    EST.queue_input(
                        (color_filepath, depth_filepath, skel_filepath,
                         intr_mat, False))

                    (filered_skel, prep_time, infer_time, track_time, _
                     ) = EST.queue_output()

                    status = EST.PE.display(win_name=dev,
                                            speed=display_speed,
                                            scale=args.op_display,
                                            image=None,
                                            bounding_box=False,
                                            tracks=EST.TK.tracks)
                    if not status[0]:
                        break_loop = True

                    # 8. printout ----------------------------------------------
                    runtime['PE'].append(1/infer_time)
                    runtime['TK'].append(1/track_time)
                    tqdm_bar.set_description(
                        f"Image : {color_filepath.split('/')[-1]} | "
                        f"#Skel filtered : {filered_skel} | "
                        f"#Tracks : {len(EST.TK.tracks)} | "
                        f"Prep time : {prep_time:.3f} | "
                        f"Pose time : {infer_time:.3f} | "
                        f"Track time : {track_time:.3f} | "
                        f"FPS PE : {sum(runtime['PE'])/len(runtime['PE']):.3f} | "  # noqa
                        f"FPS TK : {sum(runtime['TK'])/len(runtime['TK']):.3f}"
                    )

                    if args.op_rs_delete_image:
                        os.remove(color_filepath)

                if break_loop:
                    EST.break_process_loops()
                    break

            end_loop = True


if __name__ == "__main__":

    [arg_op, _] = get_parser().parse_known_args()

    extract_skel_func = rs_extract_skeletons_and_track_offline_mp

    # extract_skel_func(arg_op)

    # arg_op.op_save_result_image = True

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_deepsort")
    # arg_op.op_track_deepsort = True
    # extract_skel_func(arg_op)
    # arg_op.op_track_deepsort = False

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_bytetrack")
    arg_op.op_track_bytetrack = True
    extract_skel_func(arg_op)
    arg_op.op_track_bytetrack = False

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_ocsort")
    # arg_op.op_track_ocsort = True
    # extract_skel_func(arg_op)
    # arg_op.op_track_ocsort = False

    # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> op_track_strongsort")
    # arg_op.op_track_strongsort = True
    # extract_skel_func(arg_op)
    # arg_op.op_track_strongsort = False

    print(f"[INFO] : FINISHED")
