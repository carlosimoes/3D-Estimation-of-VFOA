import os,argparse
from sys import path
from Depth_Estimator import Depth_Estimator
dir = "/home/carlos/Thesis_GOD/detectron2/"


parser = argparse.ArgumentParser()
parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
#parser.add_argument('-in',help="path of the video to analyse",default= dir + "Thesis_GOD/Final_videos/got.mp4")
#? Video Rect
parser.add_argument('-d',help="path of the video rect", default= dir + "Thesis_GOD/Final_videos/mpi_sintel2.mp4") # videoshort.mp4")
#? Stereo videos
'''
parser.add_argument('-dvL',help="path of the Stereo video to analyse left camera", default= dir + "Thesis_GOD/Final_videos/Dataset/"+"dataset52_l.mp4")
parser.add_argument('-dvR',help="path of the Stereo video to analyse right camera", default= dir + "Thesis_GOD/Final_videos/Dataset/"+"dataset52_r.mp4")
parser.add_argument('-dvC',help="path to import camera matrix of the Stereo video to analyse", default= dir + "Thesis_GOD/Final_videos/Dataset/"+"dataset52_l.mp4")
'''
parser.add_argument('-dvL',help="path of the Stereo video to analyse left camera", default= dir + "Thesis_GOD/Final_videos/"+"stereo/training/clean_left/bandage_2")
parser.add_argument('-dvR',help="path of the Stereo video to analyse right camera", default= dir + "Thesis_GOD/Final_videos/"+"stereo/training/clean_right/bandage_2")
parser.add_argument('-dvC',help="path to import camera matrix of the Stereo video to analyse", default= dir + "Thesis_GOD/Final_videos/"+"stereo/training/camdata_left/bandage_2")
#? Dir
parser.add_argument('-od',help="path to directory where we want to create the final folder",default= dir +"Thesis_GOD/Final_videos")
#? Video for Gaze and Features Vizualization are written
parser.add_argument('-w',help="boolean, if true videos for features vizualization are written",default=True)
parser.add_argument('-g',help="boolean, if true gaze is estimated after head boxes extraction.",default=True)
#? Cuda was on index number '0' -> nvidia-smi [terminal] in case of multiple devices update this
parser.add_argument('-dev', help="GPUs devices for model inference, separated by commas",default="") 
#? Features that were not matched are not stored in the final map results
parser.add_argument('-m',help="boolean, TRUE if set keypoints and head boxes are matched to gather instances data.",default=False) 
#? len of Video
parser.add_argument('-b',help="video frames batch size to be read and processed together",default=20000,type=int) 
#? Pose Estimation BackBone
parser.add_argument('-cfg', help='experiment configure file name',
                        default=dir+'Pose_Estimation/experiments/vgg19_368x368_sgd.yaml',
                        type=str)
#? Model for Pose Estimation
parser.add_argument('-opM', type=str,
                        default=dir+'Pose_Estimation/pose_model.pth')
#? Detectron Estimation BackBone -> https://github.com/facebookresearch/detectron2/blob/master/projects/DensePose/doc/MODEL_ZOO.md
parser.add_argument('-dpCfg',help="path of the densepose model config file",default=dir+"projects/DensePose/configs/"+"densepose_rcnn_R_50_FPN_s1x_legacy.yaml") # densepose_rcnn_R_101_FPN_s1x.yaml
# densepose_rcnn_R_50_FPN_s1x.yaml
#? Model for Detectron Estimation
parser.add_argument('-dpM',help="path to the densepose model file",default=dir+"projects/DensePose/configs/"+"model_final_d366fa.pkl") #DensePose_ResNet101_FPN_s1x-e2e.pkl
#? Panoptic Estimation BackBone
parser.add_argument('-PpCfg',help="path of the Panoptic model config file",default=dir+"configs/COCO-PanopticSegmentation/"+"panoptic_fpn_R_101_3x.yaml") # "panoptic_fpn_R_50_3x.yaml"
#? Model for Panoptic Estimation
parser.add_argument('-PpM',help="path to the Panoptic model file",default=dir+"configs/COCO-PanopticSegmentation/"+"model_final_cafdb1.pkl") #  "model_final_c10459.pkl"
#? Model for Gaze360
parser.add_argument('-gM',help="path to the gaze360 model file",default=dir+"gaze360/gaze360_model.pth.tar")
#? batch sizes
parser.add_argument('-PpB', help="Panoptic model batch size",default=1, type=int)
parser.add_argument('-opB', help="openpose model batch size", default=10, type=int)
parser.add_argument('-dpB', help="Densepose model batch size",default=1, type=int)
parser.add_argument('-gB', help="gaze360 model batch size",default=10, type=int)
#? thresholds
parser.add_argument('-PpT',help="Panoptic model threshold, between 0 and 1 : the greater, the lesser number of detected instances are returned",default=0.5, type=float)
parser.add_argument('-dpT',help="Densepose model threshold, between 0 and 1 : the greater, the lesser number of detected instances are returned",default=0.85, type=float)
parser.add_argument('-gT', help="Gaze model uncertainty threshold : the greater, more estimated gazes are returned",default=3.0, type=float)
#? Model for Head Pose Estimation
parser.add_argument('-hM1', help="Head model1",default="/home/carlos/Thesis_GOD/detectron2/headpose-fsanet-pytorch/pretrained/fsanet-1x1-iter-688590.onnx")
parser.add_argument('-hM2', help="Head model2",default="/home/carlos/Thesis_GOD/detectron2/headpose-fsanet-pytorch/pretrained/fsanet-var-iter-688590.onnx")
args = parser.parse_args()
begin, nFrames = 0,249

from VideoProcessor import VideoProcessor


# Environment variable must be set before torch initialization, otherwise has no effect
# os.environ["CUDA_VISIBLE_DEVICES"] = args.dev 

#print("process file")
#viz = VideoProcessor(args)
#viz.processVideo(inputvideo=args.d, imagesBatchLength=args.b, writeVideo=args.w, extractGaze=args.g) 


if __name__ == "__main__":
    if os.path.exists(args.dvL) and os.path.exists(args.dvR):
        print("Stereo Videos to analyse exist")   
        depth=Depth_Estimator(args.dvL,args.dvR, args.dvC,"mpi_sintel")
        matrix_Q, output_disparity=depth(args.d, 0, 51 )
        del depth
    else:
        print("Stereo Videos does not exist")
    if os.path.exists(args.d):
        print("Video to analyse exist")
        if os.path.isfile(args.d):
            print("Process file", args.d)
            videoName = args.d[args.d.rfind('/'):-4]
            if os.path.exists(args.od + videoName):
                print("File previously processed, data erased for new processing")
                for entryRaw in os.scandir(args.od + videoName):
                    os.remove(entryRaw.path)
                os.rmdir(args.od + videoName)
            print("New processing of the Video")
            viz = VideoProcessor(args, matrix_Q, disp=output_disparity) #matrix_Q= None, disp= None) #matrix_Q, disp=output_disparity)
            os.mkdir(args.od + videoName)
            viz.processVideo(args.d, imagesBatchLength=args.b, extractGaze=args.g, writeVideo=args.w, matchPointsBox=args.m)
        else:
            print("Video to analyse doesnt exist!")
    else:
        print("File doesnt exist!")
    
