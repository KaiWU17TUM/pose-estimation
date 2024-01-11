import argparse
import json
import time
from datetime import datetime
import os
import numpy as np

import pyrealsense2 as rs
import cv2 as cv
import pyopenpose as op


from rs_py.utility import printout

def findDevices():
    ctx = rs.context()  # Create librealsense context for managing devices
    serials = []
    if (len(ctx.devices) > 0):
        for dev in ctx.devices:
            print('Found device: ', dev.get_info(rs.camera_info.name), ' ', dev.get_info(rs.camera_info.serial_number))
            serials.append(dev.get_info(rs.camera_info.serial_number))
    else:
        printout("No Intel Device connected", 'e')

    return serials, ctx


def enableDevices_and_saveCalib(save_path, trial, serials, ctx, resolution_width=848, resolution_height=480, frame_rate=5):
    pipelines = []
    for serial in serials:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        # enable color and depth streaming
        cfg.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
        cfg.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        profile = pipe.start(cfg) # Start pipeline and get the configuration it found
        pipelines.append([serial, pipe])

        #save calib data
        profile_rgb = profile.get_stream(rs.stream.color)
        intr_rgb = profile_rgb.as_video_stream_profile().get_intrinsics()
        intr_rgb_mat = [intr_rgb.fx, 0, intr_rgb.ppx,
                        0, intr_rgb.fy, intr_rgb.ppy,
                        0, 0, 1]
        profile_depth = profile.get_stream(rs.stream.depth)  # Fetch stream profile for depth stream
        intr_depth = profile_depth.as_video_stream_profile().get_intrinsics()  # Downcast to video_stream_profile and fetch intrinsics
        intr_depth_mat = [intr_depth.fx, 0, intr_depth.ppx,
                          0, intr_depth.fy, intr_depth.ppy,
                          0, 0, 1]

        # Extrinsic matrix from RGB sensor to Depth sensor
        extr = profile_rgb.as_video_stream_profile().get_extrinsics_to(profile_depth)
        extr_mat = np.eye(4)
        extr_mat[:3, :3] = np.array(extr.rotation).reshape(3, 3)
        extr_mat[:3, 3] = extr.translation

        # Depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_sensor = rs.depth_stereo_sensor(depth_sensor)
        depth_scale = depth_sensor.get_depth_scale()
        depth_baseline = depth_sensor.get_stereo_baseline()
        # print("Depth Scale is: ", depth_scale)

        # Write calibration data to json file
        calib_data = {}
        calib_data['rgb'] = []
        calib_data['rgb'].append({
            'width': intr_rgb.width,
            'height': intr_rgb.height,
            'intrinsic_mat': intr_rgb_mat,
            'model': str(intr_rgb.model),
            'coeffs': intr_rgb.coeffs,
            'format': 'rgb8',
            'fps': frame_rate,
        })
        calib_data['depth'] = []
        calib_data['depth'].append({
            'width': intr_depth.width,
            'height': intr_depth.height,
            'intrinsic_mat': intr_depth_mat,
            'model': str(intr_depth.model),
            'coeffs': intr_depth.coeffs,
            'depth_scale': depth_scale,
            'depth_baseline': depth_baseline,
            'format': 'z16',
            'fps': frame_rate,
        })
        calib_data['T_rgb_depth'] = []
        calib_data['T_rgb_depth'].append({
            'rotation': extr.rotation,
            'translation': extr.translation
        })

        device_path = os.path.join(save_path, serial, trial)
        calib_path = os.path.join(device_path, 'calib')
        filename = f'dev{serial}_calib.json'
        file_path = os.path.join(calib_path, filename)

        with open(file_path, 'w') as outfile:
            json.dump(calib_data, outfile, indent=4)

        printout(f'Realsense calibration data saved to {save_path}', 'i')

    return pipelines


def pipelineStop(pipelines):
    for (device,pipe) in pipelines:
        # Stop streaming
        pipe.stop()



def create_save_folders(save_path, serials, trial):
    for device_sn in serials:
        device_path = os.path.join(save_path, device_sn, trial)
        # timestamp
        timestamp_path = os.path.join(device_path, 'timestamp')
        os.makedirs(timestamp_path, exist_ok=True)
        # calib
        calib_path = os.path.join(device_path, 'calib')
        os.makedirs(calib_path, exist_ok=True)
        # color
        color_path = os.path.join(device_path, 'color')
        color_metadata_path = os.path.join(device_path, 'color_metadata')
        os.makedirs(color_path, exist_ok=True)
        os.makedirs(color_metadata_path, exist_ok=True)
        # depth
        depth_path = os.path.join(device_path, 'depth')
        depth_metadata_path = os.path.join(device_path, 'depth_metadata')
        os.makedirs(depth_path, exist_ok=True)
        os.makedirs(depth_metadata_path, exist_ok=True)
        # skeleton
        skeleton_path = os.path.join(device_path, 'skeleton')
        os.makedirs(skeleton_path, exist_ok=True)
        printout("Prepared storage paths...", 'i')


def get_op_config(op_net_resolution="-1x400"):
    params = dict()
    params["model_folder"] = "/usr/local/src/openpose/models/"
    params["model_pose"] = "BODY_25"
    params["net_resolution"] = op_net_resolution

    params["disable_blending"] = True
    params["scale_number"] = 1
    params["body"] = 1

    return params

# def save_record_timestamps(save_path, device_sn, trial, internal_timestamp, frame_dict):
#     ts_file = os.path.join(os.path.join(save_path, device_sn, trial), 'timestamp.txt')
#     with open(ts_file, 'a+') as f:
#         f.write(f"{internal_timestamp}::"
#                 f"{frame_dict['color_timestamp']}::"
#                 f"{frame_dict['depth_timestamp']}\n")

def save_2d_skeleton(keypoints,
                     scores,
                     save_path):
    # keypoints: [M, V, C]; C = (x,y,score)
    # scores: [M]
    save_path = os.path.join(save_path, 'skeleton')
    if keypoints is None:
        open(save_path, 'a').close()
    else:
        M, _, _ = keypoints.shape
        data = np.concatenate([scores.reshape((M, 1)),
                               keypoints.reshape((M, -1))], axis=1)
        np.savetxt(save_path, data, delimiter=',')



if __name__=='__main__':
    trial_time = datetime.now().strftime("%y%m%d%H%M%S")

    p = argparse.ArgumentParser(description='Run OPENPOSE on RS images')
    p.add_argument('--save-path',
                   type=str,
                   default="",
                   help='path to saved camera data')
    p.add_argument('--fps',
                   type=int,
                   default=5,
                   help='frame per second')

    args = p.parse_args()
    save_path = args.save_path

    # set up realsense cameras
    img_width = 848  # pixels
    img_height = 480  # pixels
    frame_rate = p.fps  # fps

    serials, ctx = findDevices()
    create_save_folders(save_path, serials, trial_time)
    pipelines = enableDevices_and_saveCalib(save_path, trial_time, serials, ctx, img_width, img_height, frame_rate)
    align_to = rs.stream.color
    align = rs.align(align_to)


    # # set up openpose extractor
    # op_net_resolution = "-1x400"
    # params = get_op_config(op_net_resolution)
    #
    # opWrappers = []
    # for i in range(2):
    #     wrapper = op.WrapperPython()
    #     wrapper.configure(params)
    #     wrapper.start()
    #     opWrappers.append(wrapper)

    if len(serials) != 2:
        printout('Only accept 2 cameras!', 'e')

    c = 0
    frame_ts_prev = 0
    try:
        while True:
            # for (serial, profile), opWrapper in zip(pipelines, opWrappers):
            for (serial, profile) in pipelines:
                frame_ts = time.time_ns()

                # Get frameset of color and depth
                frames = profile.wait_for_frames()
                # Align the depth frame to color frame
                aligned_frames = align.process(frames)
                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # datum = op.Datum()
                # datum.cvInputData = color_image
                # opWrapper.emplaceAndPop(op.VectorDatum([datum]))
                # skeletons = datum.poseKeypoints
                # poseScores = datum.poseScores
                #
                # output_img = datum.cvOutputData

                # save data
                depth_path = os.path.join(save_path, serial, trial_time, 'depth')
                color_path = os.path.join(save_path, serial, trial_time, 'color')
                np.save(os.path.join(depth_path, str(frame_ts)), depth_image)
                cv.imwrite(os.path.join(color_path, str(frame_ts) + '.png'), color_image)

                #save frame_ts
                with open(os.path.join(save_path, serial, trial_time, 'timestamp', 'timestamps.txt'), 'w+') as timestamp_file:
                    timestamp_file.write(f"{frame_ts}\r")

            if c % 15 == 0:
                print(f'FRAME: {c}\t---\t{1/((frame_ts - frame_ts_prev) / 1e9)}\t---\t{frame_ts}')

            frame_ts_prev = frame_ts
            c += 1


                # # Render images
                # depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
                # images = np.hstack((color_image, depth_colormap))
                #
                # cv.imshow('RealSense' + device, images)
                # key = cv.waitKey(1)
                # # Press esc or 'q' to close the image window
                # if key & 0xFF == ord('q') or key == 27:
                #     cv.destroyAllWindows()
                #     return True
                #
                # # Save images and depth maps from both cameras by pressing 's'
                # if key == 115:
                #     cv.imwrite(str(device) + '_aligned_depth.png', depth_image)
                #     cv.imwrite(str(device) + '_aligned_color.png', color_image)
                #     print('Save')

            # if exit == True:
            #     print('Program closing...')
            #     break
    finally:
        pipelineStop(pipelines)
        print('Stop realsense piplines.')
