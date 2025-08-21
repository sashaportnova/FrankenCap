# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 15:31:11 2025

@authors: Sasha Portnova and Ally Clarke

This is the run file to perform upper-body tracking with lower-limbs occluded 
by the mobility device using two front-facing cameras. It requires an intrinsics 
and an extrinsics videos for camera calibration.
"""

import main

runMarkerAugmentation=True
scaleModel=True
runInverseKinematics=True
subj='P003'
fps=30              # Frame rate of your recording device
startTime = 0       # This is if you need to trim the original video to
                    # only perform upper-body kinematic tracking for a portion of the recording
endTime = 60        # you can leave them as None if you want to keep the original
trialName = 'play'

main.main(subj, startTime, endTime, trialName,
         runMarkerAugmentation=runMarkerAugmentation, scaleModel=scaleModel, 
         runInverseKinematics=runInverseKinematics,fps=fps)
