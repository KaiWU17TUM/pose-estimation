import cv2
import json
import numpy as np
import os
import pyopenpose as op

from tqdm import tqdm
from time import sleep


class PyOpenPoseNative(object):

    def __init__(self, params: dict = None) -> None:
        super().__init__()

        # default parameters
        if params is None:
            params = dict()
            params["model_folder"] = "/usr/local/src/openpose/models/"
            params["model_pose"] = "BODY_25"
            params["net_resolution"] = "-1x368"

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.datum = op.Datum()

    def configure(self, params: dict = None):
        if params is not None:
            self.opWrapper.configure(params)

    def initialize(self):
        # Starting OpenPose
        self.opWrapper.start()

    def predict(self, image):
        self.datum.cvInputData = image
        self.opWrapper.emplaceAndPop(
            op.VectorDatum([self.datum])
        )

    @property
    def opencv_image(self):
        return self.datum.cvOutputData

    @property
    def pose_scores(self) -> list:
        return self.datum.poseScores

    @property
    def pose_keypoints(self) -> list:
        return self.datum.poseKeypoints


def read_realsense_config(file_path):

    class RealsenseConfig:
        def __init__(self, json_file):
            self.width = json_file['rgb'][0]['width']
            self.height = json_file['rgb'][0]['height']
            self.rgb_intrinsics = np.array(json_file['rgb'][0]['intrinsic_mat']).reshape(3, 3)  # noqa
            self.depth_intrinsics = np.array(json_file['depth'][0]['intrinsic_mat']).reshape(3, 3)  # noqa
            self.depth_scale = json_file['depth'][0]['depth_scale']
            self.T_rgb_depth = np.eye(4)
            self.T_rgb_depth[:3, :3] = np.array(json_file['T_rgb_depth'][0]['rotation']).reshape(3, 3)  # noqa
            self.T_rgb_depth[:3, 3] = json_file['T_rgb_depth'][0]['translation']  # noqa

    with open(file_path) as calib_file:
        calib = json.load(calib_file)
    return RealsenseConfig(calib)


def get_3d_skeleton(skeleton, depth_img, intr_mat):
    patch_offset = 2
    H, W = depth_img.shape
    fx = intr_mat[0, 0]
    fy = intr_mat[1, 1]
    cx = intr_mat[0, 2]
    cy = intr_mat[1, 2]
    joints3d = []
    for x, y, _ in skeleton:
        patch = depth_img[
            max(0, int(y-patch_offset)):min(H, int(y+patch_offset)),  # noqa
            max(0, int(x-patch_offset)):min(W, int(x+patch_offset))  # noqa
        ]
        depth_avg = np.mean(patch)
        x3d = (x-cx) / fx * depth_avg
        y3d = (y-cy) / fy * depth_avg
        joints3d.append([x3d, y3d, depth_avg])
    return np.array(joints3d)


if __name__ == "__main__":

    CLIPS_PATH = "/DigitalICU/ICRA2022/data/all/clips"

    params = dict()
    params["model_folder"] = "/usr/local/src/openpose/models/"
    params["model_pose"] = "BODY_25"
    params["net_resolution"] = "-1x368"

    pyop = PyOpenPoseNative(params)
    pyop.initialize()

    for clip_id in sorted(os.listdir(CLIPS_PATH)):
        clip_path = os.path.join(CLIPS_PATH, clip_id)

        print("Processing :", clip_path)

        calib_path = os.path.join(clip_path, 'calib.txt')
        calib_data = read_realsense_config(calib_path)

        rgb_path = os.path.join(clip_path, 'rgb')
        depth_path = os.path.join(clip_path, 'depth')

        os.makedirs(os.path.join(clip_path, 'skeleton'), exist_ok=True)
        assert not os.listdir(os.path.join(clip_path, 'skeleton'))

        kpt_arr, skel_arr = None, None

        for rgb_file in tqdm(sorted(os.listdir(rgb_path))):

            rgb_file_path = os.path.join(rgb_path, rgb_file)
            # rgb_file_path = '/home/chen/openpose/pexels-photo-4384679.jpeg'
            rgb_img = cv2.imread(rgb_file_path)
            # rgb_img = cv2.resize(rgb_img, (368, 368))

            pyop.predict(rgb_img)

            scores = pyop.pose_scores
            max_score_idx = np.argmax(scores)

            keypoint = pyop.pose_keypoints[max_score_idx]
            keypoint = np.expand_dims(keypoint, axis=0)
            if kpt_arr is None:
                kpt_arr = np.copy(keypoint)
            else:
                kpt_arr = np.append(kpt_arr, keypoint, axis=0)

            keypoint_image = pyop.opencv_image
            cv2.putText(keypoint_image,
                        "KP (%) : " + str(round(max(scores), 2)),
                        (10, 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA)
            skeleton_rgb_path = rgb_file_path.replace("rgb", "skeleton_rgb")
            skeleton_rgb_path = skeleton_rgb_path[:-4] + ".jpg"
            cv2.imwrite(skeleton_rgb_path, keypoint_image)

            depth_file_path = os.path.join(depth_path, rgb_file)
            depth_img = cv2.imread(depth_file_path, -1)

            skeleton3d = get_3d_skeleton(keypoint[0],
                                         depth_img,
                                         calib_data.rgb_intrinsics)
            skeleton3d = np.expand_dims(skeleton3d, axis=0)
            if skel_arr is None:
                skel_arr = np.copy(skeleton3d)
            else:
                skel_arr = np.append(skel_arr, skeleton3d, axis=0)

            # print("Skeleton 3D :", skeleton3d)

            sleep(0.01)

        npy_path = os.path.join(clip_path, 'skeleton.npy')
        np.save(npy_path, keypoint)
        npy_path = os.path.join(clip_path, 'skeleton_3d.npy')
        np.save(npy_path, skeleton3d)

    # Display Image
    # print("Body keypoints: \n" + str(pyop.pose_keypoints))
    # print("Prediction score: \n" + str(pyop.pose_scores))
    # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", keypoint_image)
    # cv2.waitKey(0)