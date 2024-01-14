import argparse
from .utils import str2bool


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Extract skeleton using OPENPOSE')

    # OPEN POSE OPTIONS --------------------------------------------------------
    p.add_argument('--op-mode',
                   type=str,
                   default="online",
                   help='what mode to run the inference. [single,online,offline,comparetracker]')  # noqa

    p.add_argument('--op-model-folder',
                   type=str,
                   default="/usr/local/src/openpose/models/",
                   help='folder with trained openpose models.')
    p.add_argument('--op-model-pose',
                   type=str,
                   default="BODY_25",
                   help='pose model name')
    p.add_argument('--op-net-resolution',
                   type=str,
                   default="-1x368",
                   help='resolution of input to openpose.')
    p.add_argument('--op-skel-thres',
                   type=float,
                   default=0.2,
                   help='threshold for valid skeleton.')
    p.add_argument('--op-max-true-body',
                   type=int,
                   default=8,
                   help='max number of skeletons to save.')
    p.add_argument('--op-heatmaps-add-parts',
                   type=str2bool,
                   default=False,
                   help='')
    p.add_argument('--op-heatmaps-add-bkg',
                   type=str2bool,
                   default=False,
                   help='')
    p.add_argument('--op-heatmaps-add-PAFs',
                   type=str2bool,
                   default=False,
                   help='')
    p.add_argument('--op-heatmaps-scale',
                   type=int,
                   default=1,
                   help='')
    p.add_argument('--op-save-skel-folder',
                   type=str,
                   default="",
                   help='folder to save skeleton results.')
    p.add_argument('--op-save-skel',
                   type=str2bool,
                   default=False,
                   help='if true, saves the 2d skeleton.')
    p.add_argument('--op-save-skel-image',
                   type=str2bool,
                   default=False,
                   help='if true, saves the 2d skeleton image.')
    # p.add_argument('--op-skel-file',
    #                type=str,
    #                default="skel.txt",
    #                help='file to save skeleton results.')
    p.add_argument('--op-input-color-image',
                   type=str,
                   default="",
                   help='path to input color image/folder.')
    p.add_argument('--op-image-width',
                   type=int,
                   default=1920,
                   help='image width in px')
    p.add_argument('--op-image-height',
                   type=int,
                   default=1080,
                   help='image height in px')

    # DEPTH OPTIONS ------------------------------------------------------------
    p.add_argument('--op-patch-offset',
                   type=int,
                   default=2,
                   help='offset of patch used to determine depth')
    p.add_argument('--op-extract-3d-skel',
                   type=str2bool,
                   default=False,
                   help='if true, tries to extract 3d skeleton.')
    p.add_argument('--op-save-3d-skel',
                   type=str2bool,
                   default=False,
                   help='if true, saves the 3d skeleton.')

    # DISPLAY OPTIONS ----------------------------------------------------------
    p.add_argument('--op-display',
                   type=float,
                   default=1.0,
                   help='scale for displaying skel images.')
    p.add_argument('--op-display-depth',  # NO USED
                   type=int,
                   default=0,
                   help='scale for displaying skel images with depth.')

    # REALSENSE OPTIONS --------------------------------------------------------
    p.add_argument('--op-rs-dir',
                   type=str,
                   default='data/mot17',
                   help='path to folder with saved rs data.')
    # p.add_argument('--op-rs-save-heatmaps',
    #                type=str2bool,
    #                default=False,
    #                help='if true, saves the heatmaps/output of network.')
    p.add_argument('--op-rs-delete-image',
                   type=str2bool,
                   default=False,
                   help='if true, deletes the rs image used in inference.')

    # TRACKING OPTIONS ---------------------------------------------------------
    p.add_argument('--op-track-deepsort',
                   type=str2bool,
                   default=False,
                   help='If true performs deepsort tracking.')
    p.add_argument('--op-track-bytetrack',
                   type=str2bool,
                   default=False,
                   help='If true performs ByteTrack tracking.')
    p.add_argument('--op-track-ocsort',
                   type=str2bool,
                   default=False,
                   help='If true performs OC Sort tracking.')
    p.add_argument('--op-track-strongsort',
                   type=str2bool,
                   default=False,
                   help='If true performs StrongSort tracking.')
    p.add_argument('--op-track-buffer',
                   type=int,
                   default=30,
                   help="the frames for keep lost tracks.")

    p.add_argument('--deepsort-metric', type=str, default='cosine')
    p.add_argument('--deepsort-opt-nsa', type=str2bool, default=False)
    p.add_argument('--deepsort-opt-ema', type=str2bool, default=False)
    p.add_argument('--deepsort-opt-emaalpha', type=float, default=0.0)
    p.add_argument('--deepsort-opt-mc', type=str2bool, default=False)
    p.add_argument('--deepsort-opt-mclambda', type=float, default=0.0)
    # without cascade in linear assignment
    p.add_argument('--deepsort-opt-woc', type=str2bool, default=False)
    p.add_argument('--deepsort-opt-maxcosinedistance', type=float, default=0.2)
    p.add_argument('--deepsort-opt-nnbudget', type=int, default=100)
    p.add_argument('--strongsort-metric', type=str, default='cosine')
    p.add_argument('--strongsort-opt-nsa', type=str2bool, default=True)
    p.add_argument('--strongsort-opt-ema', type=str2bool, default=True)
    p.add_argument('--strongsort-opt-emaalpha', type=float, default=0.9)
    p.add_argument('--strongsort-opt-mc', type=str2bool, default=True)
    p.add_argument('--strongsort-opt-mclambda', type=float, default=0.98)
    # without cascade in linear assignment
    p.add_argument('--strongsort-opt-woc', type=str2bool, default=True)
    p.add_argument('--bytetracker-trackthresh', type=float, default=0.5)
    p.add_argument('--bytetracker-trackbuffer', type=int, default=30)
    p.add_argument('--bytetracker-matchthresh', type=float, default=0.8)
    p.add_argument('--bytetracker-mot20', type=str2bool, default=False)
    p.add_argument('--bytetracker-fps', type=float, default=30)
    p.add_argument('--ocsort-detthresh', type=float, default=0.5)
    p.add_argument('--ocsort-maxage', type=int, default=30)
    p.add_argument('--ocsort-minhits', type=int, default=3)
    p.add_argument('--ocsort-iouthreshold', type=float, default=0.3)
    p.add_argument('--ocsort-deltat', type=int, default=3)
    p.add_argument('--ocsort-assofunc', type=str, default="iou")
    p.add_argument('--ocsort-inertia', type=float, default=0.2)
    p.add_argument('--ocsort-usebyte', type=str2bool, default=True)

    # EXPERIMENTS OPTIONS ------------------------------------------------------
    # the 2d pose can be saved with --op-save-skel-image
    p.add_argument('--op-save-track-image',
                   type=str2bool,
                   default=False,
                   help='If true saves tracking+pose results in image form.')
    p.add_argument('--op-proc',
                   type=str,
                   default='sp',
                   help='multi (mp) or single processing (sp) or none. Based on Alphapose')  # noqa

    return p
