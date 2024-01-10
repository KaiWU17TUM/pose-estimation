import argparse
import os
import time
import traceback
import sys
import numpy as np
from queue import Queue
from threading import Thread

from rs_py.rs_run_devices import printout
from rs_py.rs_run_devices import RealsenseWrapper

from openpose.native.python.skeleton import OpenPosePoseExtractor


class PoseExtractionMultithreading:

    def __init__(self, args) -> None:
        self.CQ = [Queue(maxsize=10) for _ in range(3)]
        self.PE1 = OpenPosePoseExtractor(args)
        self.PE1.pyop.configure(params=self.PE1.pyop.params)
        self.PE2 = OpenPosePoseExtractor(args)
        self.PE2.pyop.configure(params=self.PE2.pyop.params)
        self.PE3 = OpenPosePoseExtractor(args)
        self.PE3.pyop.configure(params=self.PE3.pyop.params)
        self.CP1 = Thread(target=self.predict1, args=())
        self.CP1.start()
        self.CP2 = Thread(target=self.predict2, args=())
        self.CP2.start()
        self.CP3 = Thread(target=self.predict3, args=())
        self.CP3.start()

    def predict1(self):
        while True:
            (image, save_path, break_loop) = self.CQ[0].get()
            if break_loop:
                break
            self.PE1.predict(image, None, save_path)

    def predict2(self):
        while True:
            (image, save_path, break_loop) = self.CQ[1].get()
            if break_loop:
                break
            self.PE2.predict(image, None, save_path)

    def predict3(self):
        while True:
            (image, save_path, break_loop) = self.CQ[2].get()
            if break_loop:
                break
            self.PE3.predict(image, None, save_path)


class PoseHeatMapExtractionMultithreading:

    def __init__(self, args) -> None:
        self.CQ = [Queue(maxsize=10) for _ in range(3)]
        self.PE1 = OpenPosePoseExtractor(args)
        self.PE1.pyop.configure(params=self.PE1.pyop.params_cout)
        self.PE2 = OpenPosePoseExtractor(args)
        self.PE2.pyop.configure(params=self.PE2.pyop.params_cout)
        self.PE3 = OpenPosePoseExtractor(args)
        self.PE3.pyop.configure(params=self.PE3.pyop.params_cout)
        self.CP1 = Thread(target=self.predict_hm1, args=())
        self.CP1.start()
        self.CP2 = Thread(target=self.predict_hm2, args=())
        self.CP2.start()
        self.CP3 = Thread(target=self.predict_hm3, args=())
        self.CP3.start()

    def predict_hm1(self):
        while True:
            (image, save_path, break_loop) = self.CQ[0].get()
            if break_loop:
                break
            self.PE1.predict_hm(image, save_path)

    def predict_hm2(self):
        while True:
            (image, save_path, break_loop) = self.CQ[1].get()
            if break_loop:
                break
            self.PE2.predict_hm(image, save_path)

    def predict_hm3(self):
        while True:
            (image, save_path, break_loop) = self.CQ[2].get()
            if break_loop:
                break
            self.PE3.predict_hm(image, save_path)


def extract_pose_from_heatmaps(base_path: str,
                               op_args: argparse.Namespace):

    PE = OpenPosePoseExtractor(op_args)
    PE.pyop.configure(PE.pyop.params_cin)

    devices = os.listdir(base_path)
    trials = os.listdir(os.path.join(base_path, devices[0]))

    for device in devices:
        for trial in trials:

            path = f"{device}/{trial}/skeleton_fromheatmap"
            path = os.path.join(base_path, path)
            os.makedirs(path, exist_ok=True)

            path = f"{device}/{trial}/heatmap"
            path = os.path.join(base_path, path)
            hm_files = [os.path.join(path, i) for i in os.listdir(path)]

            for hm_file in hm_files:
                hm = np.fromfile(hm_file, np.float32)
                hm = [hm[5:].reshape(int(hm[1]), int(
                    hm[2]), int(hm[3]), int(hm[4]))]
                PE.predict_from_hm(
                    image=np.zeros((480, 848, 3)),
                    heatmap=hm,
                    kpt_save_path=hm_file.replace(
                        "/heatmap", "/skeleton_fromheatmap"
                    ).replace(".float", ".txt")
                )
                PE.display(win_name=device, speed=1)

    printout(f"Finished...", 'i')


def save_heatmaps(rs_args: argparse.Namespace,
                  op_args: argparse.Namespace):

    # PE = MPPE(op_args)
    PE = PoseHeatMapExtractionMultithreading(op_args)

    RSW = RealsenseWrapper(rs_args, rs_args.rs_dev)
    RSW.initialize_depth_sensor_ae()

    if len(RSW.enabled_devices) == 0:
        raise ValueError("no devices connected")

    RSW = RealsenseWrapper(rs_args, rs_args.rs_dev)
    RSW.initialize()

    if rs_args.rs_save_data:
        RSW.storage_paths.create()
        RSW.save_calib()

    RSW.flush_frames(rs_args.rs_fps * 3)
    time.sleep(3)

    device_sns = list(RSW.enabled_devices.keys())
    timer = []

    hm_dirs = []
    for device_sn in device_sns:
        hm_dirs.append(
            RSW.storage_paths.color[device_sn].replace("/color", "/heatmap")
        )
        RSW.storage_paths.color[device_sn] = None

    try:
        c = 0

        while True:

            start = time.time()
            RSW.step(
                display=0,
                display_and_save_with_key=rs_args.rs_save_with_key,
                use_colorizer=False
            )

            # images = [RSW.frames[device_sn]['color_framedata']
            #           for device_sn in device_sns]
            # images = np.concatenate(images, axis=1)
            # PE.CQ[0].put(
            #     (images,
            #      os.path.join(
            #          hm_dirs[0],
            #          str(RSW.internal_timestamp[device_sns[0]])) + ".float",
            #      False)
            # )

            for idx, device_sn in enumerate(device_sns):
                PE.CQ[idx].put(
                    (RSW.frames[device_sn]['color_framedata'],
                     os.path.join(
                        hm_dirs[idx],
                        str(RSW.internal_timestamp[device_sn])) + ".float",
                     False)
                )

            timer.append(time.time() - start)

            if c % rs_args.rs_fps == 0:
                printout(
                    f"Step {c:12d} :: "
                    f"{len(timer)//sum(timer)} :: "
                    f"{[i.get('color_timestamp', None) for i in RSW.frames.values()]} :: "  # noqa
                    f"{[i.get('depth_timestamp', None) for i in RSW.frames.values()]}",  # noqa
                    'i'
                )
                timer = []

            if not len(RSW.frames) > 0:
                printout(f"Empty...", 'w')
                continue

            c += 1
#            if c > rs_args.rs_steps:
#                break

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        printout(f"{exc_type}, {fname}, {exc_tb.tb_lineno}", 'e')
        printout(f"Exception msg : {e}", 'e')
        traceback.print_tb(exc_tb)
        printout(f"Stopping RealSense devices...", 'i')
        RSW.stop()

    finally:
        printout(f"Final RealSense devices...", 'i')
        RSW.stop()

    for i in range(3):
        PE.CQ[i].put((None, None, True))

    printout(f"Finished...", 'i')
