# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 09:10:10 2025

@author: Sasha Portnova

This file contains everything to run OpenPose pose detection algorithm via the 
command line (taken from the OpenCap code).
"""

import os
import cv2
import shutil
import pickle
import numpy as np
import json
import sys
import time

import subprocess

from decouple import config

def runOpenPoseCMD(pathOpenPose, resolutionPoseDetection, cameraDirectory,
                   fileName, openposeJsonDir, pathOutputVideo, trialPrefix, 
                   generateVideo, videoFullPath, pathOutputJsons):
    
    horizontal = True  #meaning the video orientation is horizontal
    
    command = None
    if resolutionPoseDetection == 'default':
        cmd_hr = ' '
    elif resolutionPoseDetection == '1x1008_4scales':
        if horizontal:
            cmd_hr = ' --net_resolution "1008x-1" --scale_number 4 --scale_gap 0.25 '
        else:
            cmd_hr = ' --net_resolution "-1x1008" --scale_number 4 --scale_gap 0.25 '
    elif resolutionPoseDetection == '1x736':
        if horizontal:
            cmd_hr = ' --net_resolution "736x-1" '
        else:
            cmd_hr = ' --net_resolution "-1x736" '  
    elif resolutionPoseDetection == '1x736_2scales':
        if horizontal:
            cmd_hr = ' --net_resolution "-1x736" --scale_number 2 --scale_gap 0.75 '
        else:
            cmd_hr = ' --net_resolution "736x-1" --scale_number 2 --scale_gap 0.75 '
        
    if config("DOCKERCOMPOSE", cast=bool, default=False):
        vid_path_tmp = "/data/tmp-video.mov"
        vid_path = "/data/video_openpose.mov"
        
        # copy the video to vid_path_tmp
        shutil.copy(f"{cameraDirectory}/{fileName}", vid_path_tmp)
        
        # rename the video to vid_path
        os.rename(vid_path_tmp, vid_path)

        try:
            # wait until the video is processed (i.e. until the video is removed -- then json should be ready)
            start = time.time()
            while True:
                if not os.path.isfile(vid_path):
                    break
                
                if start + 60*60 < time.time():
                    raise Exception("Pose detection timed out. This is unlikely to be your fault, please report this issue on the forum. You can proceed with your data collection (videos are uploaded to the server) and later reprocess errored trials.", 'timeout - openpose')
                
                time.sleep(0.1)
            
            # copy /data/output to openposeJsonDir
            os.system("cp /data/output_openpose/* {cameraDirectory}/{openposeJsonDir}/".format(cameraDirectory=cameraDirectory, openposeJsonDir=openposeJsonDir))
        
        except Exception as e:
            if len(e.args) == 2: # specific exception
                raise Exception(e.args[0], e.args[1])
            elif len(e.args) == 1: # generic exception
                exception = "Pose detection failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed neutral pose."
                raise Exception(exception, exception)   
            
    elif pathOpenPose == "docker":
        
        command = "docker run --gpus=1 -v {}:/openpose/data stanfordnmbl/openpose-gpu\
            /openpose/build/examples/openpose/openpose.bin\
            --video /openpose/data/{}\
            --display 0\
            --write_json /openpose/data/{}\
            --render_pose 0{}".format(cameraDirectory, fileName,
                                        openposeJsonDir, cmd_hr)
    else:
        os.chdir(pathOpenPose)
        pathVideoOut = os.path.join(pathOutputVideo,
                                    trialPrefix + 'withKeypoints.avi')
        if not generateVideo:
            command = ('bin\\OpenPoseDemo.exe --video "{}" --write_json "{}" '
                       '--render_threshold 0.5 --display 0 --render_pose 0{}').format(
                       videoFullPath, pathOutputJsons, cmd_hr)
        else:
            command = ('bin\\OpenPoseDemo.exe --video "{}" --write_json "{}" '
                       '--render_threshold 0.5 --display 0{} --write_video "{}"').format(
                       videoFullPath, pathOutputJsons, cmd_hr, pathVideoOut)

    if command:
        os.system(command)
    
    return