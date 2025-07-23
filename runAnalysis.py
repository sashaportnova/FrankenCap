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
subj='P002'
fps=30
startTime = 0
endTime = 60
trialName = 'play2'

main.main(subj, startTime, endTime, trialName,
         runMarkerAugmentation=runMarkerAugmentation, scaleModel=scaleModel, 
         runInverseKinematics=runInverseKinematics,fps=30)
