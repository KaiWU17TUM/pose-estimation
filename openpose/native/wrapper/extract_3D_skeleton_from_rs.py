import argparse
import cv2
import numpy as np
import os

from datetime import datetime

from skeleton import get_3d_skeleton
from skeleton import PyOpenPoseNative
from realsense import RealsenseWrapper
from realsense import StoragePaths


class OpenposeStoragePaths(StoragePaths):
    def __init__(self, device_sn: str = ''):
        super().__init__()
        base_path = '/data/openpose'
        date_time = datetime.now().strftime("%y%m%d%H%M%S")
        self.calib = f'{base_path}/calib/dev{device_sn}_{date_time}'
        self.color = f'{base_path}/color/dev{device_sn}_{date_time}'
        self.depth = f'{base_path}/depth/dev{device_sn}_{date_time}'
        self.skeleton = f'{base_path}/skeleton/dev{device_sn}_{date_time}'
        self.timestamp = f'{base_path}/timestamp/dev{device_sn}_{date_time}'
        self.timestamp_file = os.path.join(self.timestamp, 'timestamp.txt')
        os.makedirs(self.calib, exist_ok=True)
        os.makedirs(self.color, exist_ok=True)
        os.makedirs(self.depth, exist_ok=True)
        os.makedirs(self.skeleton, exist_ok=True)
        os.makedirs(self.timestamp, exist_ok=True)


def save_skeleton_3d(skeleton_3d: np.ndarray, skeleton_save_path: str) -> None:
    skeleton_3d_str = ",".join(
        [str(pos) for skel in skeleton_3d.tolist() for pos in skel])
    with open(skeleton_save_path, 'a+') as f:
        f.write(f'{skeleton_3d_str}\n')


def save_skel(pyop: PyOpenPoseNative,
              arg: argparse.Namespace,
              depth_image: np.ndarray,
              intr_mat: np.ndarray,
              empty_skeleton_3d: np.ndarray,
              skeleton_save_path: str,
              ) -> None:
    scores = pyop.pose_scores

    # 3.a. Save empty array if scores is None (no skeleton at all)
    if scores is None:
        for _ in range(arg.max_true_body):
            save_skeleton_3d(empty_skeleton_3d, skeleton_save_path)
        print("No skeleton detected...")

    else:
        # 3.b. Save prediction scores
        # max_score_idx = np.argmax(scores)
        max_score_idxs = np.argsort(scores)[-arg.max_true_body:]

        if arg.save_skel:
            for max_score_idx in max_score_idxs:

                if scores[max_score_idx] < arg.save_skel_thres:
                    save_skeleton_3d(empty_skeleton_3d, skeleton_save_path)
                    print("Low skeleton score, skip skeleton...")

                else:
                    keypoint = pyop.pose_keypoints[max_score_idx]
                    # ntu_format => x,y(up),z(neg) in meter.
                    skeleton_3d = get_3d_skeleton(
                        skeleton=keypoint,
                        depth_img=depth_image,
                        intr_mat=intr_mat,  # noqa
                        ntu_format=arg.ntu_format
                    )
                    save_skeleton_3d(skeleton_3d, skeleton_save_path)

            for _ in range(arg.max_true_body-len(max_score_idxs)):
                save_skeleton_3d(empty_skeleton_3d, skeleton_save_path)


def display_skel(pyop: PyOpenPoseNative, device_sn: str) -> bool:
    keypoint_image = pyop.opencv_image
    cv2.putText(keypoint_image,
                "KP (%) : " + str(round(max(pyop.pose_scores), 2)),
                (10, 20),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                1,
                cv2.LINE_AA)
    cv2.imshow(f'keypoint_image_{device_sn}', keypoint_image)
    key = cv2.waitKey(30)
    # Press esc or 'q' to close the image window
    if key & 0xFF == ord('q') or key == 27:
        cv2.destroyAllWindows()
        cv2.waitKey(5)
        return False
    else:
        return True


def get_parser():
    parser = argparse.ArgumentParser(
        description='Extract 3D skeleton using OPENPOSE')

    parser.add_argument('--fps',
                        type=int,
                        default=30,
                        help='fps')

    parser.add_argument('--display-rs',
                        type=bool,
                        default=False,
                        help='if true, display realsense raw images.')
    parser.add_argument('--display-skel',
                        type=bool,
                        default=False,
                        help='if true, display skel images from openpose.')

    parser.add_argument('--save-skel',
                        default=True,
                        help='if true, save 3d skeletons.')
    parser.add_argument('--save-skel-thres',
                        type=float,
                        default=0.5,
                        help='threshold for valid skeleton.')
    parser.add_argument('--max-true-body',
                        type=int,
                        default=2,
                        help='max number of skeletons to save.')

    parser.add_argument('--model-folder',
                        type=str,
                        default="/usr/local/src/openpose/models/",
                        help='foilder with trained openpose models.')
    parser.add_argument('--model-pose',
                        type=str,
                        default="BODY_25",
                        help=' ')
    parser.add_argument('--net-resolution',
                        type=str,
                        default="-1x368",
                        help='resolution of input to openpose.')
    parser.add_argument('--ntu-format',
                        type=bool,
                        default=False,
                        help='whether to use coordinate system of NTU')

    return parser


if __name__ == "__main__":

    arg = get_parser().parse_args()

    state = True
    empty_skeleton_3d = np.zeros((25, 3))

    # 0. Initialize ------------------------------------------------------------
    # OPENPOSE
    params = dict(
        model_folder=arg.model_folder,
        model_pose=arg.model_pose,
        net_resolution=arg.net_resolution,
    )
    pyop = PyOpenPoseNative(params)
    pyop.initialize()

    # REALSENSE # STORAGE
    rsw = RealsenseWrapper()
    rsw.stream_config.fps = 30
    rsw.initialize()
    rsw.set_storage_paths(OpenposeStoragePaths)
    rsw.save_calibration()

    try:
        while state:

            # 1. Get rs data ---------------------------------------------------
            frames = rsw.run(display=arg.display_rs)
            if not len(frames) > 0:
                continue

            for dev_sn, data_dict in frames.items():
                color_image = frames[dev_sn]['color']
                depth_image = frames[dev_sn]['depth']
                timestamp = frames[dev_sn]['timestamp']
                calib = frames[dev_sn]['calib']

                # 2. Predict pose ----------------------------------------------
                # bgr format
                pyop.predict(color_image)

                # 3. Save data -------------------------------------------------
                if arg.save_skel:
                    intr_mat = calib['color'][0]['intrinsic_mat']
                    skel_save_path = os.path.join(
                        rsw.storage_paths[dev_sn].skeleton,
                        f'{timestamp:020d}' + '.txt'
                    )
                    save_skel(pyop, arg, depth_image, intr_mat,
                              empty_skeleton_3d, skel_save_path)

                if arg.display_skel:
                    state = display_skel(pyop, dev_sn)

    except:  # noqa
        print("Stopping realsense...")
        rsw.stop()

    finally:
        rsw.stop()
