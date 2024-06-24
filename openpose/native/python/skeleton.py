import argparse
import cv2
import os
import numpy as np
import pyopenpose as op

from typing import Optional, Union, List, Tuple, Any
from .utils import get_color


TORSO_IDS = [1, 2, 5, 8, 9, 12]

# IMG = cv2.imread(
#     "/usr/local/src/openpose/examples/media/COCO_val2014_000000000192.jpg")


class PyOpenPoseNativeBase:
    """Base wrapper for the native openpose python interface. This base class
    contains the properties and utililty functions for the main class.
    """

    def __init__(self,
                 skel_thres: float = 0.0,
                 max_true_body: int = 2,
                 patch_offset: int = 2) -> None:
        super().__init__()
        # for 3d skel
        self.skel_thres = skel_thres
        self.max_true_body = max_true_body
        self.patch_offset = patch_offset
        # main data class from pyopenpose
        self.datum = op.Datum()
        # results
        self._pose_empty = False
        self._pose_heatmaps = None
        self._pose_keypoints_3d = None
        self._pose_bounding_box = None
        self._pose_bounding_box_int = None
        self.reset()

    def reset(self) -> None:
        self._pose_empty = False
        self._pose_heatmaps = None
        self._pose_keypoints_3d = None
        self._pose_bounding_box = None
        self._pose_bounding_box_int = None
        return

    @property
    def opencv_image(self) -> np.ndarray:
        # [H, W, C]
        return self.datum.cvOutputData

    @property
    def pose_empty(self):
        return self._pose_empty

    @property
    def pose_scores(self) -> Optional[np.ndarray]:
        # [M]
        return self.datum.poseScores

    @property
    def pose_heatmaps(self) -> Optional[np.ndarray]:
        # [op_h, op_w, 76] : all heatmaps
        # [M, J, 3] : person, joints, xyz
        if self._pose_heatmaps is None:
            if self.datum.poseHeatMaps is None:
                j = self.datum.poseKeypoints.shape[1]
                w = self.datum.netInputSizes[0].x
                h = self.datum.netInputSizes[0].y
                self._pose_heatmaps = np.zeros((h, w, j))
            else:
                self._pose_heatmaps = np.moveaxis(self.datum.poseHeatMaps,
                                                  0, -1)
        return self._pose_heatmaps

    @property
    def pose_keypoints(self) -> Optional[np.ndarray]:
        # [M, V, C]; M = Subjects; V = Joints; C = (x,y,score)
        return self.datum.poseKeypoints

    @property
    def pose_keypoints_3d(self) -> Optional[np.ndarray]:
        # [M, V, C]; C = (x,y,z)
        return self._pose_keypoints_3d

    @property
    def pose_bounding_box(self) -> Optional[np.ndarray]:
        if self._pose_empty:
            return None
        if self._pose_bounding_box is None:
            bb = []

            for pose_keypoints in self.pose_keypoints:
                u = pose_keypoints[TORSO_IDS, 0]
                v = pose_keypoints[TORSO_IDS, 1]
                s = pose_keypoints[TORSO_IDS, 2]
                if (s != 0).any():
                    u_min = u[s != 0].min()
                    u_max = u[s != 0].max()
                    v_min = v[s != 0].min()
                    v_max = v[s != 0].max()
                    if u_min == u_max:
                        u_max += 1
                    if v_min == v_max:
                        v_max += 1
                else:
                    u_min = v_min = 0
                    u_max = v_max = 1
                bb.append(np.asarray([u_min, v_min, u_max, v_max]))
            self._pose_bounding_box = np.stack(bb)
        return self._pose_bounding_box

    @property
    def pose_bounding_box_int(self) -> Optional[np.ndarray]:
        if self._pose_empty:
            return None
        if self._pose_bounding_box_int is None:
            _bb = self.pose_bounding_box
            if _bb is not None:
                u_min = np.floor(_bb[:, 0]).astype(int)
                v_min = np.floor(_bb[:, 1]).astype(int)
                u_max = np.ceil(_bb[:, 2]).astype(int)
                v_max = np.ceil(_bb[:, 3]).astype(int)
                self._pose_bounding_box_int = np.stack(
                    [u_min, v_min, u_max, v_max], axis=1)
        return self._pose_bounding_box_int

    @staticmethod
    def get_3d_skeleton(skeleton: np.ndarray,
                        depth_img: np.ndarray,
                        intr_mat: Union[list, np.ndarray],
                        depth_scale: float = 1e-3,
                        patch_offset: int = 2) -> np.ndarray:
        if isinstance(intr_mat, list):
            fx = intr_mat[0]
            fy = intr_mat[4]
            cx = intr_mat[2]
            cy = intr_mat[5]
        elif isinstance(intr_mat, np.ndarray):
            fx = intr_mat[0, 0]
            fy = intr_mat[1, 1]
            cx = intr_mat[0, 2]
            cy = intr_mat[1, 2]
        else:
            raise ValueError("Unknown intr_mat format.")
        H, W = depth_img.shape
        joints3d = []
        for x, y, _ in skeleton:
            patch = depth_img[
                max(0, int(y-patch_offset)):min(H, int(y+patch_offset)),
                max(0, int(x-patch_offset)):min(W, int(x+patch_offset))
            ]
            depth_avg = np.nanmean(patch)
            x3d = (x-cx) / fx * depth_avg
            y3d = (y-cy) / fy * depth_avg
            joints3d.append([x3d*depth_scale, y3d*depth_scale,
                            depth_avg*depth_scale])
        return np.array(joints3d)

    def convert_to_3d(self,
                      depth_image: np.ndarray,
                      intr_mat: Union[list, np.ndarray],
                      depth_scale: float = 1e-3,
                      ) -> None:
        if self.pose_empty:
            # print("No skeleton detected...")
            pass
        else:
            pose_keypoints_3d = []
            # Empty array if scores is None (no skeleton at all)
            for pose_keypoint in self.pose_keypoints:
                # [V,C]
                skeleton_3d = self.get_3d_skeleton(
                    skeleton=pose_keypoint,
                    depth_img=depth_image,
                    intr_mat=intr_mat,
                    depth_scale=depth_scale,
                    patch_offset=self.patch_offset
                )
                pose_keypoints_3d.append(skeleton_3d)
            self._pose_keypoints_3d = np.asarray(pose_keypoints_3d)
        return

    @staticmethod
    def save_2d_skeleton(keypoints: Optional[np.ndarray],
                         scores: Optional[np.ndarray],
                         save_path: str) -> None:
        # keypoints: [M, V, C]; C = (x,y,score)
        # scores: [M]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if keypoints is None:
            open(save_path, 'a').close()
        else:
            M, _, _ = keypoints.shape
            data = np.concatenate([scores.reshape((M, 1)),
                                   keypoints.reshape((M, -1))], axis=1)
            np.savetxt(save_path, data, delimiter=',')

    def save_pose_keypoints(self, save_path: str) -> None:
        self.save_2d_skeleton(keypoints=self.pose_keypoints,
                              scores=self.pose_scores,
                              save_path=save_path)
        return

    def save_3d_pose_keypoints(self, save_path: str) -> None:
        # keypoints: [M, V, C]; C = (x,y,z)
        # scores: [M]
        self.save_2d_skeleton(keypoints=self.pose_keypoints_3d,
                              scores=self.pose_scores,
                              save_path=save_path)
        return

    @staticmethod
    def _draw_text_on_skeleton_image(image: Optional[np.ndarray] = None,
                                     scores: Optional[np.ndarray] = None
                                     ) -> np.ndarray:
        if scores is not None:
            image = cv2.putText(image,
                                "KP (%) : " + str(round(np.mean(scores), 2)),
                                (10, 50),
                                cv2.FONT_HERSHEY_PLAIN,
                                1,
                                (0, 255, 0),
                                1,
                                cv2.LINE_AA)
        return image

    @staticmethod
    def _draw_depth_on_skeleton_image(image: Optional[np.ndarray] = None,
                                      depth: Optional[np.ndarray] = None
                                      ) -> np.ndarray:
        if depth is not None:
            colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth, alpha=0.065, beta=0),
                cv2.COLORMAP_INFERNO
            )
            # colormap = cv2.flip(colormap, 1)
            image = cv2.addWeighted(
                image, 0.7, colormap, 0.7, 0)
            # overlay = cv2.resize(overlay, (800, 450))
        return image

    @staticmethod
    def _draw_bounding_box_on_skeleton_image(
            image: np.ndarray,
            boxes: Optional[np.ndarray]) -> np.ndarray:
        if boxes is not None:
            for idx, bb in enumerate(boxes):
                tl, br = bb[0:2], bb[2:4]
                image = cv2.rectangle(image, tl, br, (0, 125, 0), 2)
        return image

    @staticmethod
    def _draw_tracking_bounding_box_image(
            image: np.ndarray,
            tracks: Optional[list] = None) -> np.ndarray:
        if tracks is not None:
            for track in tracks:
                if track.is_activated:
                    try:
                        # deepsort / ocsort
                        bb = track.to_tlbr()
                    except AttributeError:
                        # bytetrack
                        bb = track.tlbr
                    try:
                        l, t, r, b = bb
                        tl = (np.floor(l).astype(int), np.floor(t).astype(int))
                        br = (np.ceil(r).astype(int), np.ceil(b).astype(int))
                        (x1, y1), (x2, y2) = tl, br
                        sub_img = image[y1:y2, x1:x2]
                        rect = (np.ones(sub_img.shape, dtype=np.uint8) *
                                # np.array(get_color(track.track_id), dtype=np.uint8))  # noqa
                                np.array((60, 130, 0), dtype=np.uint8))
                        res = cv2.addWeighted(sub_img, 0.5, rect, 0.5, 1.0)
                        image[y1:y2, x1:x2] = res
                        # image = cv2.rectangle(image, tl, br,
                        #                       get_color(track.track_id), 3)
                        cv2.putText(image,
                                    f"ID : {track.track_id}",
                                    tl,
                                    cv2.FONT_HERSHEY_PLAIN,
                                    2,
                                    get_color(track.track_id),
                                    2,
                                    cv2.LINE_AA)
                    except TypeError:
                        pass
        return image

    def display(self,
                win_name: str,
                speed: int = 1000,
                scale: float = 1.0,
                depth_image: Optional[np.ndarray] = None,
                bounding_box: bool = False,
                tracks: Optional[list] = None,
                ori_image: Optional[np.ndarray] = None
                ) -> Tuple[bool, Optional[np.ndarray]]:
        """Displays an image with results.

        Args:
            win_name (str): Name of the cv window.
            speed (int, optional): for cv2.waitKey. Defaults to 1000.
            scale (float, optional): sacle of the image to display.
                Defaults to 1.0.
            depth_image (Optional[np.ndarray], optional): Depth image.
                Defaults to None.
            bounding_box (bool, optional): Whether to draw skeleton bb.
                Defaults to False.
            tracks (Optional[list], optional): Track class to draw the
                track ids. Defaults to None.
            ori_image (Optional[np.ndarray], optional): Original image to be
                concatenaated alongside the image with results if given.
                Defaults to None.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (Bool whether to close the
                cv window, image shown in cv window)
        """

        image = self.opencv_image

        image = self._draw_depth_on_skeleton_image(image, depth_image)

        # image = self._draw_text_on_skeleton_image(image, self.pose_scores)

        if bounding_box:
            image = self._draw_bounding_box_on_skeleton_image(
                image, self.pose_bounding_box_int)

        if tracks is not None:
            image = self._draw_tracking_bounding_box_image(image, tracks)

        image = cv2.resize(image, (int(image.shape[1]*scale),
                                   int(image.shape[0]*scale)))

        if ori_image is not None:
            _image = cv2.resize(ori_image, (int(ori_image.shape[1]*scale),
                                            int(ori_image.shape[0]*scale)))
            image = np.concatenate([_image, image], axis=0)

        if scale >= 1000:
            cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(win_name,
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        # cv2.imshow(win_name, image)
        # key = cv2.waitKey(speed)
        # # Press esc or 'q' to close the image window
        # if key & 0xFF == ord('q') or key == 27:
        #     cv2.destroyAllWindows()
        #     cv2.waitKey(5)
        #     return False, image
        # else:
        #     return True, image
        return True, image

    def save_skeleton_image(self,
                            save_path: str,
                            depth: Optional[np.ndarray] = None) -> None:
        image = self.opencv_image
        image = self._draw_depth_on_skeleton_image(image, depth)
        if not self.pose_empty:
            image = self._draw_text_on_skeleton_image(image, self.pose_scores)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, image)
        return


class PyOpenPoseNative(PyOpenPoseNativeBase):
    """Wrapper for the native openpose python interface. """

    def __init__(self,
                 params: Optional[dict] = None,
                 skel_thres: float = 0.0,
                 max_true_body: int = 2,
                 patch_offset: int = 2,
                 ) -> None:
        super().__init__(skel_thres, max_true_body, patch_offset)

        # default parameters
        if params is None:
            params = dict()
            params["model_folder"] = "/usr/local/src/openpose/models/"
            params["model_pose"] = "BODY_25"
            params["net_resolution"] = "-1x368"
            # params["heatmaps_add_parts"] = True
            # params["heatmaps_add_bkg"] = True
            # params["heatmaps_add_PAFs"] = True
            # params["heatmaps_scale"] = 2
            # params["upsampling_ratio"] = 1  # for saving raw heatmaps

        params["disable_blending"] = True

        params["scale_number"] = 1
        params["body"] = 1


        # params["posenet_only"] = False
        # params["custom_net_input_layer"] = ""
        # params["custom_net_output_layer"] = ""
        self.params = params.copy()

        # Get heatmap from certain layer in caffe model
        params["scale_number"] = 1
        params["body"] = 1
        # params["posenet_only"] = True
        # saves heatmaps in unscaled size. Not used if posenet_only=True
        # params["upsampling_ratio"] = 1
        # params["custom_net_input_layer"] = ""
        # params["custom_net_output_layer"] = "pool3_stage1"
        self.params_cout = params.copy()

        # Get keypoints from heatmaps
        params["scale_number"] = 1
        params["body"] = 1  # 2 to Disable OP Network
        # params["posenet_only"] = False
        # saves heatmaps in unscaled size. Not used if posenet_only=True
        # params["upsampling_ratio"] = 0  # 0 rescales to input image size
        # params["custom_net_input_layer"] = "pool3_stage1"
        # params["custom_net_output_layer"] = ""
        self.params_cin = params.copy()

        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()

        # results
        self.filtered_skel = "0"
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.filtered_skel = "0"
        return

    def configure(self, params: dict = None) -> None:
        if params is not None:
            self.opWrapper.stop()
            self.opWrapper.configure(params)
            self.opWrapper.start()
        return

    def predict(self, image: np.ndarray) -> None:
        self.reset()
        self.datum.cvInputData = image
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        if self.datum.poseScores is None:
            self._pose_empty = True
        return

    def predict_hm(self,
                   image: np.ndarray,
                   hm_save_path: str) -> None:
        self.reset()
        self.datum.cvInputData = image
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        os.makedirs(os.path.dirname(hm_save_path), exist_ok=True)
        self.datum.poseRawHeatMaps[0].tofile(hm_save_path)
        np.insert(self.datum.poseRawHeatMaps[0].reshape(-1),
                  [0, 0, 0, 0, 0],
                  [self.datum.poseRawHeatMaps[0].ndim] +
                  list(self.datum.poseRawHeatMaps[0].shape)
                  ).tofile(hm_save_path)
        if self.datum.poseScores is None:
            self._pose_empty = True
        return

    def predict_from_hm(self,
                        image: np.ndarray,
                        heatmap: Optional[List[np.ndarray]] = None,) -> None:
        self.reset()
        self.datum.cvInputData = image
        self.datum.customInputNetData = heatmap
        # self.configure(self.params_cin)
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        self.datum.customInputNetData = []

        if self.datum.poseScores is None:
            self._pose_empty = True
        return

    def filter_prediction(self) -> None:
        # 1. Empty array if scores is None (no skeleton at all)
        if self.pose_empty:
            # print("No skeleton detected...")
            pass
        # 2. Else pick pose based on prediction scores
        else:
            scores = self.pose_scores
            scores_filtered = []
            keypoints_filtered = []
            max_score_idxs = np.argsort(scores)[-self.max_true_body:]
            for max_score_idx in max_score_idxs:
                if scores[max_score_idx] < self.skel_thres:
                    # print(f"Low skeleton score {scores[max_score_idx]:.2f}, "
                    #       f"skip skeleton...")
                    continue
                else:
                    keypoint = self.pose_keypoints[max_score_idx]
                    keypoints_filtered.append(keypoint)
                    scores_filtered.append(scores[max_score_idx])
            if len(scores_filtered) == 0:
                self.reset()
            else:
                # [M, V, C]; M = Subjects; V = Joints; C = (x,y,score)
                self._pose_keypoints = np.stack(keypoints_filtered, axis=0)
                # [M]
                self._pose_scores = np.stack(scores_filtered, axis=0)

            self.filtered_skel = f"{len(keypoints_filtered)}/{len(scores)}"  # noqa
            # print(f"Skeletons filtered: {len(keypoints_filtered)}/{len(scores)}")  # noqa

        return


class OpenPosePoseExtractor:
    """A high level wrapper for the `PyOpenPoseNative` class. This class only
    has functions that intialize openpose, infer + save pose, and display pose.
    """

    def __init__(self, args: argparse.Namespace, face: bool) -> None:
        self.pyop = PyOpenPoseNative(
            dict(
                model_folder=args.op_model_folder,
                model_pose=args.op_model_pose,
                net_resolution=args.op_net_resolution,
                heatmaps_add_parts=args.op_heatmaps_add_parts,
                heatmaps_add_bkg=args.op_heatmaps_add_bkg,
                heatmaps_add_PAFs=args.op_heatmaps_add_PAFs,
                heatmaps_scale=args.op_heatmaps_scale,
                render_pose=-1 if args.op_display > 0 else 0,
                face=face
            ),
            args.op_skel_thres,
            args.op_max_true_body,
            args.op_patch_offset,
        )

    def predict(self,
                image: np.ndarray,
                depth: np.ndarray,
                kpt_save_path: Optional[str] = None,
                skel_image_save_path: Optional[str] = None) -> None:
        self.pyop.predict(image)
        self.pyop.filter_prediction()
        if kpt_save_path is not None:
            self.pyop.save_pose_keypoints(kpt_save_path)
        if skel_image_save_path is not None:
            self.pyop.save_skeleton_image(skel_image_save_path, depth)

    def predict_hm(self,
                   image: np.ndarray,
                   hm_save_path: str) -> None:
        self.pyop.predict_hm(image, hm_save_path)

    def predict_from_hm(self,
                        image: np.ndarray,
                        heatmap: Optional[List[np.ndarray]] = None,
                        kpt_save_path: Optional[str] = None) -> None:
        self.pyop.predict_from_hm(image, heatmap)
        self.pyop.filter_prediction()
        if kpt_save_path is not None:
            self.pyop.save_pose_keypoints(kpt_save_path)

    def predict_3d(self,
                   image: np.ndarray,
                   depth: np.ndarray,
                   intr_mat: np.ndarray,
                   depth_scale: float = 1e-3,
                   kpt_save_path: Optional[str] = None,
                   kpt_3d_save_path: Optional[str] = None,
                   skel_image_save_path: Optional[str] = None) -> None:
        self.predict(image, depth, kpt_save_path, skel_image_save_path)
        self.pyop.convert_to_3d(
            depth_image=depth,
            intr_mat=intr_mat,
            depth_scale=depth_scale
        )
        if kpt_3d_save_path is not None:
            self.pyop.save_3d_pose_keypoints(kpt_3d_save_path)

    def display(self,
                win_name: str = "1",
                speed: int = 1000,
                scale: int = 1.0,
                image: Optional[np.ndarray] = None,
                bounding_box: bool = False,
                tracks: list = None) -> Tuple[bool, Optional[np.ndarray]]:
        if scale > 0:
            if self.pyop.datum.poseScores is None:
                img = self.pyop.opencv_image
                if image is not None:
                    _image = cv2.resize(image, (int(image.shape[1]*scale),
                                                int(image.shape[0]*scale)))
                    img = np.concatenate([_image, img], axis=0)
                # cv2.imshow(win_name, img)
                # cv2.waitKey(speed)
                return True, img

            else:
                return self.pyop.display(win_name,
                                         scale=scale,
                                         speed=speed,
                                         bounding_box=bounding_box,
                                         tracks=tracks,
                                         ori_image=image)
        else:
            return True, None
