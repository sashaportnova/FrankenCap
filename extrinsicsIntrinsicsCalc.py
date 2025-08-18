# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 09:44:49 2025

@author: Sasha Portnova

This is a simple piece of code to extract intrinsics and extrinsics from 
cameras. 

Remember, for intrisnics calculations, you need a video of the
checkerboard to be presented to the cameras at different angles 
(the checkerboard does not have to be seen by all cameras at the same time).

For extrinsics, you need a brief video of a static checkerboard and it
must be seen by all cameras at the same time.

The extrinsics calculation relies on the code from Opencap-main repository,
such as utils.py and utilsChecker2.py

"""

import cv2
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import traceback
import sys
sys.path.append('OpenCap')

# %%
def draw_camera_axes(ax, R, t, scale=100):
    # Camera center
    cam_pos = -R.T @ t  # Convert to world coordinates
    ax.scatter(*cam_pos, color="black", s=50, label="Camera Center")

    # Axes directions
    axes = scale * np.eye(3)
    for i, color in enumerate(["red", "green", "blue"]):  # X, Y, Z
        axis_end = cam_pos + R.T @ axes[:, i]
        ax.plot(*zip(cam_pos.flatten(), axis_end.flatten()), color=color)


# %% 
def plot_camera_positions(extrinsics):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for cam in extrinsics:   
        R = extrinsics[cam]['R']
        t = extrinsics[cam]['t']
        cam_pos = -R.T @ t  # Convert from camera space to world space
        ax.scatter(cam_pos[0], cam_pos[1], cam_pos[2], label=cam, s=100)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.title("Camera Positions")
    plt.show()


# %%
def calculate_reprojection_error(obj_points, img_points, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0
    total_points = 0

    for i in range(len(obj_points)):
        # Project 3D object points into 2D
        img_points_proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

        # Compute error
        error = np.linalg.norm(img_points[i] - img_points_proj, axis=2)
        total_error += np.sum(error)
        total_points += len(obj_points[i])

    mean_error = total_error / total_points
    print(f"Mean reprojection error: {mean_error:.4f} pixels")
    return mean_error

# %%
def extract_checkerboard_frames(video_path, checkerboard_dims, output_dir, frame_sampling_rate=30):
    """
    Extract frames containing a checkerboard pattern from a video.

    Parameters:
        video_path (str): Path to the video file.
        checkerboard_dims (tuple): Dimensions of the checkerboard (inner corners: rows, cols).
        output_dir (str): Directory to save extracted frames.
        frame_sampling_rate (int): Process every nth frame.

    Returns:
        list: List of file paths to extracted frames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # check if path contians saved frames already
    if any(os.path.isfile(os.path.join(output_dir, f)) for f in os.listdir(output_dir)):
        extracted_frames = []
        
        for file in os.listdir(output_dir):
            extracted_frames.append(os.path.join(output_dir,file))
        
    else:
        cap = cv2.VideoCapture(video_path)
        # Get the total frame count
        frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        extracted_frames = []
    
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return extracted_frames
    
        print(f"Processing video: {video_path}")
    
        while cap.isOpened() and frame_count < frame_count_total:            
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Failed to read frame {frame_count}.")
                frame_count += 1
                continue
    
            if frame_count % frame_sampling_rate == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # cv2.imshow("Extracted Frame", gray)
                # cv2.waitKey(0)
                ret, corners = cv2.findChessboardCornersSB(gray, checkerboard_dims,
                                                         cv2.CALIB_CB_EXHAUSTIVE + 
                                                         cv2.CALIB_CB_ACCURACY + 
                                                         cv2.CALIB_CB_LARGER)
    
                if ret:  # Checkerboard detected
                    frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    extracted_frames.append(frame_filename)
                    print(f"Checkerboard detected: Saved {frame_filename}")
    
            frame_count += 1
    
        cap.release()
        print(f"Finished processing {video_path}. {len(extracted_frames)} frames extracted.")
    return extracted_frames

# %%
def calibrate_camera_from_images(image_files, checkerboard_dims, display=False):
    """Calibrate a single camera from images."""
    obj_points = []  # 3D points in real world space
    img_points = []  # 2D points in image plane

    objp = np.zeros((checkerboard_dims[0] * checkerboard_dims[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_dims[0], 0:checkerboard_dims[1]].T.reshape(-1, 2)

    for img_file in image_files:
        img = cv2.imread(img_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCornersSB(gray, checkerboard_dims,
                                                 cv2.CALIB_CB_EXHAUSTIVE + 
                                                 cv2.CALIB_CB_ACCURACY + 
                                                 cv2.CALIB_CB_LARGER)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)

    ret, intrinsicMat, distortion, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    
    rotation_matrix, _ = cv2.Rodrigues(rvecs[0])
    
    # mean_error = calculate_reprojection_error(obj_points, img_points, rvecs, tvecs, intrinsicMat, distortion)
    # print(mean_error)
    
    if (display):
        # Define 3D points for the axes (origin + 3 axes points)
        axis_length = 2  # Length of the axes
        axis_points_3d = np.float32([
            [0, 0, 0],        # Origin
            [axis_length, 0, 0],  # X-axis (red)
            [0, axis_length, 0],  # Y-axis (green)
            [0, 0, -axis_length]  # Z-axis (blue, points "into" the image)
        ])
        
        i = 0
        for img_file in image_files:
            image = cv2.imread(img_file)
            
            #Project 3D points to 2D image plane
            axis_points_2d, _ = cv2.projectPoints(axis_points_3d, rvecs[i], tvecs[i], intrinsicMat, distortion)
            
            # Convert to integer for drawing
            origin = tuple(axis_points_2d[0].ravel().astype(int))
            x_axis = tuple(axis_points_2d[1].ravel().astype(int))
            y_axis = tuple(axis_points_2d[2].ravel().astype(int))
            z_axis = tuple(axis_points_2d[3].ravel().astype(int))
            
            # Draw the axes
            cv2.line(image, origin, x_axis, (0, 0, 255), 3)  # X-axis in red
            cv2.line(image, origin, y_axis, (0, 255, 0), 3)  # Y-axis in green
            cv2.line(image, origin, z_axis, (255, 0, 0), 3)  # Z-axis in blue
            
            # Display the image
            cv2.imshow('Image with Axes', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            i = i+1
            
        
    
    return {
        'imageSize': gray.shape[::-1],
        'intrinsicMat': intrinsicMat,
        'distortion': distortion,
        # 'rotation_EulerAngles': [cv2.Rodrigues(rvec)[0] for rvec in rvecs]
    }

# %%
def process_videos(subject, video_paths, checkerboard_dims):
    """Process videos and extract intrinsic and extrinsic parameters."""
    results = {}
    output_dir = "extracted_frames//" + subject
    frames_extracted = {}

    for i, video_path in enumerate(video_paths):
        print(f"Processing video: {video_path}")
        
        # Extract checkerboard frames
        frame_files = extract_checkerboard_frames(
            video_path,
            checkerboard_dims,
            f"{output_dir}/camera_{i+1}"
        )
        
        frames_extracted[f'camera_{i+1}'] = frame_files
        
        # Calibrate the camera using the extracted frames
        if frame_files:
            results[f'camera_{i+1}'] = calibrate_camera_from_images(frame_files, checkerboard_dims, display=False)
        else:
            print(f"No checkerboard frames found for {video_path}")

    return frames_extracted, results

# %%
def saveCameraParameters(filename,CameraParams):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename),exist_ok=True)
    
    open_file = open(filename, "wb")
    pickle.dump(CameraParams, open_file)
    open_file.close()
    
    return True

# %%
import pickle

def read_pickle_file(file_path):
    """
    Reads and returns the content of a .pickle file.

    Parameters:
        file_path (str): Path to the .pickle file.

    Returns:
        object: The deserialized object from the .pickle file.

    Raises:
        FileNotFoundError: If the file does not exist.
        pickle.UnpicklingError: If the file is not a valid .pickle file.
    """
    if not file_path.endswith('.pickle'):
        raise ValueError(f"Expected a .pickle file, but got: {file_path}")

    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    except pickle.UnpicklingError as e:
        raise pickle.UnpicklingError(f"Failed to unpickle file {file_path}: {e}")



# %%
def main(subject, path, camList,metaData):
    '''

    Parameters
    ----------
    subject : subject 
    path : path to the folder (FrankenCap)
    camList : list of cameras
    metaData : checkerboard metadata

    Returns
    -------
    None.

    '''
    base_data_dir = os.path.join(path,subject)
      
    # Find intrinsics using the moving videos
    checkerboard_dims = (metaData['black2BlackCornersHeight_n'],
                         metaData['black2BlackCornersWidth_n'])  # Checkerboard dimensions (number of inner corners)
    square_size = metaData['squareSideLength_mm']/1000   # Size of one square in meters
    
    video_files = []
    numCameras = len(camList)
    
    # Check if intrinsics.pickle exists
    exists = [False] * numCameras
    for i,cam in enumerate(camList):
        pathIntrinsics = os.path.join(base_data_dir,'Videos',cam,'CameraParameters', 'cameraIntrinsics.pickle')
        if os.path.exists(pathIntrinsics):
            exists[i] = True
    
    if not any(exists):
        for i in range(0,numCameras):
            video = base_data_dir + '\\Videos\\Cam' + str(i+1) +'\\CameraParameters\\intrinsics.mp4'
            video_files.append(video)       
    
        frame_files, calibration_results = process_videos(subject, video_files, checkerboard_dims)
        # Save intrinsics properties into a pickle file format
        for i in range(numCameras):
            video_path = base_data_dir + '\\Videos\\Cam' + str(i+1)
            saveCameraParameters(os.path.join(video_path,'CameraParameters','cameraIntrinsics.pickle'),calibration_results[f'camera_{i+1}'])

    #%% EXTRINSICS
    
    # first check if extrinsics exist    
    CheckerBoardParams = {
            'dimensions': (
                metaData['black2BlackCornersWidth_n'],
                metaData['black2BlackCornersHeight_n']),
            'squareSize': 
                metaData['squareSideLength_mm']}     
    # Camera directories and models.
    cameraDirectories = {}
    for i,camName in enumerate(camList):
        cameraDirectories[camName] = os.path.join(base_data_dir, 'Videos',
                                                  'Cam'+str(i+1))       
    
    # Get cameras' intrinsics and extrinsics.     
    # Load parameters if saved, compute and save them if not.
    from utils import loadCameraParameters
    from utilsChecker2 import rotateIntrinsics, calcExtrinsicsFromVideo
    CamParamDict = {}
    loadedCamParams = {}
    for camName in cameraDirectories:
        camDir = cameraDirectories[camName]
        # Intrinsics ######################################################
        # Intrinsics and extrinsics already exist for this session.
        if os.path.exists(
                os.path.join(camDir,'CameraParameters',"cameraIntrinsicsExtrinsics.pickle")):
            CamParams = loadCameraParameters(
                os.path.join(camDir,'CameraParameters', "cameraIntrinsicsExtrinsics.pickle"))
            loadedCamParams[camName] = True
            
        # Extrinsics do not exist for this session.
        else:
            # Intrinsics ##################################################
            # Intrinsics directories.
            
            permIntrinsicDir = os.path.join(camDir,'CameraParameters','cameraIntrinsics.pickle')
                     
            # Intrinsics exist.
            if os.path.exists(permIntrinsicDir):
                CamParams = loadCameraParameters(permIntrinsicDir)                    
                    
            # Extrinsics ##################################################
            # extension = getVideoExtension(pathVideoWithoutExtension)
            extrinsicPath = os.path.join(camDir,'CameraParameters', 'extrinsics.mp4') 
                                          
            # Modify intrinsics if camera view is rotated
            CamParams = rotateIntrinsics(CamParams,extrinsicPath)
            alternateExtrinsics=None
            useSecondExtrinsicsSolution = (
                    alternateExtrinsics is not None and 
                    camName in alternateExtrinsics)
            
            # for 720p, imageUpsampleFactor=4 is best for small board
            try:
                CamParams = calcExtrinsicsFromVideo(
                    extrinsicPath,CamParams, CheckerBoardParams, 
                    visualize=False, imageUpsampleFactor=4,
                    useSecondExtrinsicsSolution = useSecondExtrinsicsSolution)
            except Exception as e:
                if len(e.args) == 2: # specific exception
                    raise Exception(e.args[0], e.args[1])
                elif len(e.args) == 1: # generic exception
                    exception = "Camera calibration failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about camera calibration and https://www.opencap.ai/troubleshooting for potential causes for a failed calibration."
                    raise Exception(exception, traceback.format_exc())
            loadedCamParams[camName] = False
            
   
        # Append camera parameters.
        if CamParams is not None:
            CamParamDict[camName] = CamParams.copy()
        else:
            CamParamDict[camName] = None

    # Save parameters if not existing yet.
    if not all([loadedCamParams[i] for i in loadedCamParams]):
        for camName in CamParamDict:
            saveCameraParameters(
                os.path.join(cameraDirectories[camName],'CameraParameters',
                             "cameraIntrinsicsExtrinsics.pickle"), 
                os.path.join(CamParamDict[camName],'CameraParameters'))



