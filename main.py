# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 16:09:43 2025

@authors: Sasha Portnova and Ally Clarke

This is a variation on the FrankenCap that we created with OpenPose being
the algorithm used for pose detection (instead of MediaPipe as in the 
previous version).
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from scipy.signal import savgol_filter
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import subprocess
import ffmpeg 
from scipy import stats
import pandas as pd

import glob
import json
# %%
class PoseTriangulator:
    def __init__(self, camera_matrices, dist_coeffs, extrinsics):
        """
        Initialize the pose triangulator with camera parameters.
        
        Args:
            camera_matrices: List of intrinsic camera matrices [K1, K2]
            dist_coeffs: List of distortion coefficients [d1, d2]
            extrinsics: List of extrinsic parameters [R1|t1, R2|t2] relative to world frame
            filter_type: Type of filter to use ('one_euro', 'kalman', 'savgol', 'lowpass', 'none')
            
        """
        self.camera_matrices = camera_matrices
        self.dist_coeffs = dist_coeffs
        self.extrinsics = extrinsics
        
        # Setup OpenPose
        params = dict()
        params["model_folder"] = "models/"
        params["model_pose"] = "BODY_25"  # or COCO or MPI
        params["net_resolution"] = "656x368"
        
       # Setup projection matrices
        self.projection_matrices = []
        for i in range(2):
            R = extrinsics[i][:, :3]
            t = extrinsics[i][:, 3:]
            P = camera_matrices[i] @ np.hstack((R, t))
            self.projection_matrices.append(P)
            
        # BODY_25 has 25 keypoints
        self.landmark_names = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
            "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
            "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
            "RBigToe", "RSmallToe", "RHeel"
        ]        
        
        '''
        Nose - 0
        Neck - 1
        RShoulder - 2
        RElbow - 3
        RWrist - 4
        LShoulder - 5
        LElbow - 6
        LWrist - 7
        REye - 8
        LEye - 9
        
        '''
        
# %%    
    def triangulate_point(self, points_2d, visibility, visibility_threshold=0.3):
        """
        Triangulate a 3D point from 2D points observed in two cameras.
        
        Args:
            points_2d: List of 2D points [[x1, y1], [x2, y2]]
            visibility: List of point visibility [[visibility1],[visibility2]]
            visibility_threshold: Minimum visibility threshold (default: 0.4)
            
        Returns:
            point_3d: Triangulated 3D point [X, Y, Z]
        """
        # Check visibility against threshold for both cameras
        if visibility[0] < visibility_threshold or visibility[1] < visibility_threshold:
            return np.array([np.nan, np.nan, np.nan])
        
        points_2d_homogeneous = []
        for i, point in enumerate(points_2d):
            # Skip if point is NaN (not detected)
            if np.isnan(point[0]) or np.isnan(point[1]):
                return np.array([np.nan, np.nan, np.nan])
            
            # Convert to normalized coordinates
            x, y = point
            point_2d_homogeneous = np.array([x, y, 1.0]).reshape(3, 1)
            points_2d_homogeneous.append(point_2d_homogeneous)
        
        # Triangulate using DLT algorithm
        point_4d_homogeneous = cv2.triangulatePoints(
            self.projection_matrices[0],
            self.projection_matrices[1],
            points_2d_homogeneous[0][:2],
            points_2d_homogeneous[1][:2]
        )
        
        # Convert from homogeneous to Euclidean coordinates
        point_3d = point_4d_homogeneous[:3] / point_4d_homogeneous[3]
        return point_3d.reshape(-1)
 # %%   
    def triangulate_pose(self, keypoints_list, frame):
        """
        Triangulate all pose keypoints to 3D.
        
        Args:
            keypoints_list: List of keypoints from each camera
            
        Returns:
            keypoints_3d: Array of 3D keypoints
        """
        keypoints_3d = []
        
        nFrames,nMarkers,dim = keypoints_list[0].shape
        cams = len(keypoints_list)
        
        # MediaPipe returns 33 keypoints per pose
        for j in range(nMarkers):
            points_2d = []
            visibility = []
            for i in range(2):
                # Use only x, y coordinates
                points_2d.append(keypoints_list[i][frame,j,:2])
                visibility.append(keypoints_list[i][frame,j,2])
            
            # Triangulate this keypoint
            point_3d = self.triangulate_point(points_2d,visibility)
            keypoints_3d.append(point_3d)
        
        return np.array(keypoints_3d)
    
# %%
def interpolate(point_sequence, min_keyframes=2, max_gap_frames=5, method='spline'):
    """
    Interpolate 2D or 3D marker positions when they are NaN due to low visibility.
    
    Args:
        point_sequence: Array of shape (n_frames, n_dims) containing points over time
                       where n_dims can be 2 (for 2D) or 3 (for 3D)
        min_keyframes: Minimum number of valid keyframes needed on each side (default: 2)
        max_gap_frames: Maximum number of consecutive frames to interpolate (default: 5)
        method: Interpolation method ('linear' or 'spline')
    
    Returns:
        Interpolated point sequence of shape (n_frames, n_dims)
    """
    # Import scipy for spline interpolation if needed
    if method == 'spline':
        from scipy.interpolate import CubicSpline        
    
    # Get dimensions - works for both 2D and 3D
    n_frames, n_dims = point_sequence.shape
    
    # Create a copy of the input sequence to avoid modifying the original
    interpolated_sequence = np.copy(point_sequence)
    
    # Create a mask for valid points (not NaN) - check first dimension
    valid_mask = ~np.isnan(point_sequence[:, 0])
    
    # Find sequences of invalid points that need interpolation
    invalid_segments = []
    start_idx = None
    
    for i in range(len(valid_mask)):
        if not valid_mask[i] and start_idx is None:
            # Start of a new invalid segment
            start_idx = i
        elif valid_mask[i] and start_idx is not None:
            # End of an invalid segment
            invalid_segments.append((start_idx, i-1))
            start_idx = None
    
    # Handle case where the sequence ends with invalid points
    if start_idx is not None:
        invalid_segments.append((start_idx, len(valid_mask)-1))
    
    # Process each invalid segment
    for start, end in invalid_segments:
        gap_length = end - start + 1
        
        # Skip if gap is too large
        if gap_length > max_gap_frames:
            continue
        
        # Find valid keyframes before the gap
        pre_valid_indices = []
        pre_valid_points = []
        idx = start - 1
        while len(pre_valid_indices) < min_keyframes and idx >= 0:
            if valid_mask[idx]:
                pre_valid_indices.append(idx)
                pre_valid_points.append(point_sequence[idx])
            idx -= 1
        
        # Find valid keyframes after the gap
        post_valid_indices = []
        post_valid_points = []
        idx = end + 1
        while len(post_valid_indices) < min_keyframes and idx < len(valid_mask):
            if valid_mask[idx]:
                post_valid_indices.append(idx)
                post_valid_points.append(point_sequence[idx])
            idx += 1
        
        # Skip if we don't have enough keyframes
        if len(pre_valid_indices) < min_keyframes or len(post_valid_indices) < min_keyframes:
            continue  # Leave as NaN if we can't interpolate
        
        if method == 'spline' and len(pre_valid_indices) + len(post_valid_indices) >= 4:
            # Use each axis separately for spline interpolation
            for axis in range(n_dims):  # Now works for any number of dimensions
                # Get known values (before and after the gap)
                known_indices = pre_valid_indices[::-1] + post_valid_indices
                known_values = [point_sequence[i][axis] for i in known_indices]
                
                # Create cubic spline
                cs = CubicSpline(known_indices, known_values, bc_type='natural')
                
                # Apply interpolation to the gap
                for i in range(start, end+1):
                    interpolated_sequence[i, axis] = cs(i)
        else:
            # Linear interpolation
            for i in range(start, end+1):
                if method == 'linear':
                    # Find the two closest points (one before, one after)
                    pre_idx = max([idx for idx in pre_valid_indices if idx < i])
                    post_idx = min([idx for idx in post_valid_indices if idx > i])
                    
                    # Get corresponding points
                    pre_point = point_sequence[pre_idx]
                    post_point = point_sequence[post_idx]
                    
                    # Calculate interpolation factor
                    t = (i - pre_idx) / (post_idx - pre_idx)
                    
                    # Linear interpolation - works for any number of dimensions
                    interpolated_sequence[i] = (1 - t) * pre_point + t * post_point
                    
    return interpolated_sequence
#%%
from scipy.signal import firwin, filtfilt
def low_pass_filter(data, cutoff=6, fs=30, numtaps=31):
    """
    Applies a FIR low-pass filter to 3D keypoint trajectories (NaN-tolerant).

    Args:
        data: np.ndarray of shape (nFrames, nKeypoints, 3)
        cutoff: Cutoff frequency in Hz (e.g., 6 Hz for human motion)
        fs: Sampling rate in Hz (e.g., 120 for 120 fps)
        numtaps: Length of the FIR filter (more = smoother)

    Returns:
        filtered_data: np.ndarray of same shape as input
    """
    filtered = np.copy(data)
    n_frames, n_kps, n_dims = data.shape

    # Design FIR filter
    fir_coeff = firwin(numtaps, cutoff, fs=fs)

    for kp in range(n_kps):
        for dim in range(n_dims):
            signal = data[:, kp, dim]
            smoothed = np.copy(signal)

            # Identify valid (non-NaN) values
            isnan = np.isnan(signal)
            valid_idx = np.where(~isnan)[0]

            if len(valid_idx) < numtaps:
                continue  # Not enough data to apply filter

            # Split into contiguous valid segments
            gaps = np.where(np.diff(valid_idx) > 1)[0]
            segments = np.split(valid_idx, gaps + 1)

            padlen = 3 * numtaps
            for segment in segments:
                if len(segment) <= padlen:
                    continue  # Segment too short for filtfilt
                s = segment
                segment_data = signal[s]
                smoothed[s] = filtfilt(fir_coeff, [1.0], segment_data)

            filtered[:, kp, dim] = smoothed

    return filtered
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
def write_trc_header(file, fps, landmark_names):
    """
    Write the header of a TRC file.
    
    Args:
        file: File object to write to
        fps: Frames per second
        landmark_names: List of landmark names
    """
    n_markers = len(landmark_names)
    
    # First line: File type and format
    file.write("PathFileType\t4\t(X/Y/Z)\t{}\n".format(os.path.basename(file.name)))
    
    # Second line: Data format
    file.write("DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
    
    # Third line: Placeholder values (will be updated at the end)
    file.write("{}\t{}\t{}\t{}\tm\t{}\t1\t{}\n".format(
        fps, fps, 0, n_markers, fps, 0
    ))
    
    # Fourth line: Marker names (column headers)
    file.write("Frame#\tTime\t")
    for name in landmark_names:
        file.write("{}\t\t\t".format(name))
    file.write("\n")
    
    # Fifth line: Coordinate labels
    file.write("\t\t")
    for i in range(n_markers):  # fixed typo: n*markers -> n_markers
        file.write("X\tY\tZ\t")
    file.write("\n")
# %%
def update_trc_header(file_path, num_frames):
    """
    Update the header of a TRC file with the actual number of frames.
    
    Args:
        file_path: Path to the TRC file
        num_frames: Number of frames to write
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Split the line to preserve formatting
    header_parts = lines[2].strip().split('\t')
    header_parts[2] = str(num_frames)  # NumFrames
    header_parts[7] = str(num_frames)  # OrigNumFrames
    lines[2] = '\t'.join(header_parts) + '\n'
    
    with open(file_path, 'w') as file:
        file.writelines(lines)
# %%
def write_trc_frame(file, frame_idx, timestamp, keypoints_3d):
    """
    Write a frame of data to a TRC file in meters.
    
    Args:
        file: File object to write to
        frame_idx: Frame index (1-based)
        timestamp: Timestamp in seconds
        keypoints_3d: Array of 3D keypoints (assumed to be in mm)
    """
    file.write("{}\t{:.6f}\t".format(frame_idx, timestamp))
    
    for kp in keypoints_3d:
        if np.isnan(kp[0]):
            # Missing marker
            file.write("\t\t\t")
        else:
            # Convert from mm to meters (divide by 1000) and write
            file.write("{:.6f}\t{:.6f}\t{:.6f}\t".format(
                kp[0] / 1000.0, kp[1] / 1000.0, kp[2] / 1000.0
            ))
    
    file.write("\n")
    
    
#%%
def rotate_keypoints(keypoints, rotation_matrix):
    # keypoints: (num_frames, num_keypoints, 3)
    num_frames, num_keypoints, _ = keypoints.shape
    rotated = keypoints @ rotation_matrix.T  # Apply rotation to each 3D point
    return rotated

#%% 
def get_rotation_matrix(axis='x', angle_deg=90):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")
# %%
def cut_video(input_path, output_path, start_time, end_time):
    """
    Cut a video from start_time to end_time and save it as a new file.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the cut video.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
    """
    try:
        # Use ffmpeg to cut the video
        ffmpeg.input(input_path, ss=start_time, to=end_time).output(output_path).run()
        
        print(f"Video saved to {output_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")
        
# %%
def find_longest_complete_sequence(markers_positions):
    """
    Find the longest continuous sequence of frames where all markers are visible.
    
    Parameters:
    -----------
    markers_positions : numpy.ndarray
        Array of shape (frames, markers, 3) containing 3D positions of markers across multiple frames.
        NaN values indicate markers that are not visible.
    
    Returns:
    --------
    tuple
        (start_frame, end_frame) of the longest sequence where all markers are visible.
        If no complete frames are found, returns (None, None).
    int
        Length of the longest complete sequence.
    """
    # Get dimensions
    frames, num_markers, _ = markers_positions.shape
    
    # Create mask for frames where all markers are visible
    # A frame is 'complete' if none of its markers contain NaN values
    complete_frames = np.zeros(frames, dtype=bool)
    
    for frame in range(frames):
        # Check if any marker has NaN in any coordinate
        if not np.isnan(markers_positions[frame]).any():
            complete_frames[frame] = True
    
    # Find longest sequence of consecutive True values
    if not np.any(complete_frames):
        return (None, None), 0  # No complete frames found
    
    # Initialize variables to track sequences
    longest_start = longest_end = current_start = 0
    longest_length = 0
    current_length = 0
    
    # Scan through all frames
    for frame in range(frames):
        if complete_frames[frame]:
            # If this is the start of a new sequence
            if current_length == 0:
                current_start = frame
            current_length += 1
        else:
            # End of a sequence, check if it's the longest so far
            if current_length > longest_length:
                longest_length = current_length
                longest_start = current_start
                longest_end = frame - 1
            current_length = 0
    
    # Check the last sequence if the file ends with complete frames
    if current_length > longest_length:
        longest_length = current_length
        longest_start = current_start
        longest_end = frames - 1
    
    return (longest_start, longest_end), longest_length

# %%
def filter_markers_by_velocity(marker_positions, z_threshold=2.5, nMarkerDelete=5):
    """
    Filter marker positions based on velocity z-scores.
    
    Parameters:
    -----------
    marker_traj : np.ndarray
        Array of shape (nFrames, 3) representing 3D positions of a single marker.
    z_threshold : float, optional
        Z-score threshold for identifying outlier velocities.

    Returns:
    --------
    filtered_traj : np.ndarray
        Copy of the input with outliers set to NaN.
    """
    if marker_positions.shape[0] < 2:
        return marker_positions.copy()

    # Compute velocities between frames
    velocities = np.diff(marker_positions, axis=0)  # shape: (nFrames - 1, 3)
    speeds = np.linalg.norm(velocities, axis=1)  # shape: (nFrames - 1,)
    
    # Compute z-scores
    z_scores = stats.zscore(speeds, nan_policy='omit')
    
    # Find frame indices where z-score exceeds threshold
    outlier_indices = np.where(np.abs(z_scores) > z_threshold)[0]

    # Create output and set outlier frames to NaN (with +/-2 padding)
    filtered_traj = marker_positions.copy()
    for idx in outlier_indices:
        for delta in [-nMarkerDelete, 0, nMarkerDelete]:
            frame_to_nan = idx + delta
            if 0 <= frame_to_nan < filtered_traj.shape[0]:
                filtered_traj[frame_to_nan] = np.nan

    return filtered_traj

# %%
def savePickle(filename,keypoints3D):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename),exist_ok=True)
    
    open_file = open(filename, "wb")
    pickle.dump(keypoints3D, open_file)
    open_file.close()
    
    return True

# %%
def load_openpose_keypoints(json_dir):
    """
    Load 2D keypoints (x, y, confidence) for all people across all frames.
    
    Returns:
        A list of frames, where each frame is a list of people,
        and each person is an (N, 3) array of keypoints.
    """
    # Step 1: Load frame-by-frame data
    json_files = sorted(glob.glob(os.path.join(json_dir, '*.json')))
    nFrames = len(json_files)
    
    frames = []
    for file_path in json_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        people_keypoints = []
        for person in data.get('people', []):
            keypoints_flat = person.get('pose_keypoints_2d', [])
            if keypoints_flat:
                people_keypoints.append(keypoints_flat)  # keep flat to match res shape
        frames.append(people_keypoints)

    # Step 2: Reorganize data to person-by-person across frames
    allPeople = []
    iPerson = 0
    anotherPerson = True

    while anotherPerson:
        anotherPerson = False
        res = np.full((nFrames, 75), np.nan)  # 25 keypoints × (x,y,conf)

        for c_frame, frame in enumerate(frames):
            if len(frame) > iPerson:
                keypoints = frame[iPerson]
                res[c_frame, :] = keypoints
            if len(frame) > iPerson + 1:
                anotherPerson = True  # there's at least one more person in some frame

        allPeople.append(res.copy())
        iPerson += 1

    return allPeople
# %%
# Function to plot keypoints for the current frame
from PIL import Image
import matplotlib.pyplot as plt

def plot_frame(frame, allPeople, startFrame, nFrames):
    if startFrame >= nFrames:
        raise ValueError(f"startFrame ({startFrame}) must be less than nFrames ({nFrames})")
    
    fig, ax = plt.subplots()

    # Convert NumPy frame (BGR or RGB) to PIL image if needed
    if isinstance(frame, np.ndarray):
        if frame.shape[2] == 3:  # assume RGB/BGR
            img = Image.fromarray(frame.astype(np.uint8))
        else:
            raise ValueError("Frame must have 3 color channels.")
    else:
        img = frame  # Assume it's already a PIL Image

    ax.imshow(img)

    has_people = False
    for i, person_data in enumerate(allPeople):
        if person_data.size > 0 and person_data.shape[1] == 75:
            key2D = np.full((25, nFrames, 2), np.nan)
            try:
                for j in range(25):
                    key2D[j, :, 0:2] = person_data[:, j * 3:j * 3 + 2]
                x = key2D[:, startFrame, 0]
                y = key2D[:, startFrame, 1]
                ax.scatter(x, y, label=f'Person {i}')
                has_people = True
            except Exception as e:
                print(f"Skipping person {i} due to error: {e}")
                continue
        else:
            print(f"Skipping person {i}: data shape is {person_data.shape}")

    if has_people:
        ax.legend()
    plt.show()
# %%
def get_frame_at_index(video_path, frame_index):
    """Load a single frame at the specified index"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        print(f"Error: Could not read frame {frame_index}")
        return None
#%%
def get_total_frames(video_path):
    """Get total number of frames in video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

# %%
# Function to handle frame navigation and selection
def frame_navigation(video_path, allPeople, nFrames):
    """Navigate frames and select person to track"""
    startFrame = 0
    personToTrack = None
    iterations = 0
    
    total_video_frames = get_total_frames(video_path)
    max_frame = min(nFrames, total_video_frames)
    
    print(f"Video has {total_video_frames} frames, keypoint data has {nFrames} frames")
    print(f"Will navigate up to frame {max_frame}")
    
    while startFrame < max_frame:
        iterations += 1
        print(f"Loading frame {startFrame}...")
        
        # Load single frame on demand
        frame = get_frame_at_index(video_path, startFrame)
        
        if frame is not None and frame.size > 0:
            plot_frame(frame, allPeople, startFrame, nFrames)
        else:
            print(f"Frame {startFrame} is empty or invalid. Skipping.")
            startFrame += 10
            continue
        
        # Ask the user if the person is visible
        user_input = input("Is the person you want to track visible in the frame? (y/n/q to quit): ").lower()
        
        if user_input == 'y':
            try:
                personToTrack = int(input("Select the person you want to track (0, 1, 2, etc.): "))
                if personToTrack >= len(allPeople):
                    print(f"Invalid person index. Max is {len(allPeople)-1}")
                    continue
                return personToTrack, startFrame
            except ValueError:
                print("Please enter a valid number")
                continue
                
        elif user_input == 'n':
            startFrame += 10  # Skip ahead 10 frames
            
        elif user_input == 'q':
            print("Quitting frame navigation")
            break
            
        else:
            print("Invalid input. Press 'y' to select person, 'n' to skip 10 frames, 'q' to quit.")
    
    return personToTrack, startFrame
#%%
def getOpenPoseFaceMarkers():
    
    faceMarkerNames = ['Nose', 'REye', 'LEye', 'REar', 'LEar']
    markerNames = getOpenPoseMarkerNames()
    idxFaceMarkers = [markerNames.index(i) for i in faceMarkerNames]
    
    return faceMarkerNames, idxFaceMarkers

#%%
def getOpenPoseMarkerNames():
    
    markerNames = ["Nose", "Neck", "RShoulder", "RElbow", "RWrist",
                   "LShoulder", "LElbow", "LWrist", "midHip", "RHip",
                   "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",
                   "LEye", "REar", "LEar", "LBigToe", "LSmallToe",
                   "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    
    return markerNames

#%%
def getOpenPoseMarkerLowerBodyNames():
    
    LowerBodymarkerNames = ["midHip", "RHip",
                   "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
                   "LBigToe", "LSmallToe",
                   "LHeel", "RBigToe", "RSmallToe", "RHeel"]
    markerNames = getOpenPoseMarkerNames()
    idxLBMarkers = [markerNames.index(i) for i in LowerBodymarkerNames]
    
    return LowerBodymarkerNames,idxLBMarkers
#%%
def trackKeypointBox(videoPath,bbStart,allPeople,allBoxes,dataOut,cam,frameStart = 0 ,
                     frameIncrement = 1, visualize = False, poseDetector='OpenPose',
                     badFramesBeforeStop = 0):
    
    camName = cam

    # Tracks closest keypoint bounding boxes until the box changes too much.
    bboxKey = bbStart # starting bounding box
    frameNum = frameStart

    # initiate video capture
    # Read video
    video = cv2.VideoCapture(videoPath)
    nFrames = allBoxes[0].shape[0]
    
    # Read desiredFrames.
    video.set(1, frameNum)
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        raise Exception('Cannot read video file')
        
    imageSize = (frame.shape[0],frame.shape[1])
    justStarted = True
    count = 0   
    badFrames = []

    print(f"nFrames: {nFrames}, frameStart: {frameStart}, frameIncrement: {frameIncrement}")
    if frameIncrement == 0:
        raise ValueError("frameIncrement cannot be 0")

    while frameNum > -1 and frameNum < nFrames:
        # Read a new frame
        
        if visualize:
            video.set(1, frameNum)
            ok, frame = video.read()
            if not ok:
                break
        
        # Find person closest to tracked bounding box, and fill their keypoint data
        keyBoxes = [box[frameNum] for box in allBoxes]        
        
        iPerson, bboxKey_new, samePerson = findClosestBox(bboxKey, keyBoxes, 
                                                imageSize)
        
        # We allow badFramesBeforeStop of samePerson = False to account for an
        # errant frame(s) in the pose detector. Once we reach badFramesBeforeStop,
        # we break and output to the last good frame.
        if len(badFrames) > 0 and samePerson:
            badFrames = []
            
        if not samePerson and not justStarted:
            if len(badFrames) >= badFramesBeforeStop:
                print('{}: not same person at {}'.format(camName, frameNum - frameIncrement*badFramesBeforeStop))
                # Replace the data from the badFrames with zeros
                if len(badFrames) > 1:
                    dataOut[badFrames,:] = np.zeros(len(badFrames),dataOut.shape[0])
                break     
            else:
                badFrames.append(frameNum)
                
       # Don't update the bboxKey for the badFrames
        if len(badFrames) == 0:
            bboxKey = bboxKey_new

        
        dataOut[frameNum,:] = allPeople[iPerson][frameNum,:]
        
        # Next frame 
        frameNum += frameIncrement
        justStarted = False
        
        if visualize: 
            p3 = (int(bboxKey[0]), int(bboxKey[1]))
            p4 = (int(bboxKey[0] + bboxKey[2]), int(bboxKey[1] + bboxKey[3]))
            cv2.rectangle(frame, p3, p4, (0,255,0), 2, 1)
            
            # Display result
            cv2.imshow("Tracking", frame)
            
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break
        
        count += 1
        if count % 100 == 0:  # Progress indicator
            print(f"Processing frame {frameNum}, iteration {count}")
  
    return dataOut

#%%    
def findClosestBox(bbox,keyBoxes,imageSize,iPerson=None):
    # bbox: the bbox selected from the previous frame.
    # keyBoxes: bboxes detected in the current frame.
    # imageSize: size of the image
    # iPerson: index of the person to track..   
    
    # Parameters.
    # Proportion of mean image dimensions that corners must change to be
    # considered different person
    cornerChangeThreshold = 0.7
    
    keyBoxCorners = []
    for keyBox in keyBoxes:
        keyBoxCorners.append(np.array([keyBox[0], keyBox[1], 
                                       keyBox[0] + keyBox[2],
                                       keyBox[1] + keyBox[3]]))
    bboxCorners = np.array(
        [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
    
    boxErrors = [
        np.linalg.norm(keyBox - bboxCorners) for keyBox in keyBoxCorners]
    try:
        if iPerson is None:
            iPerson = np.nanargmin(boxErrors)
        bbox = keyBoxes[iPerson]
    except:
        return None, None, False
    
    # If large jump in bounding box, break.
    samePerson = True
    if (boxErrors[iPerson] > cornerChangeThreshold*np.mean(imageSize)):
        samePerson = False
        
    return iPerson,bbox,samePerson

#%%
def keypointsToBoundingBox(data,confidenceThreshold=0.3):
    # input: nFrames x 75.
    # output: nFrames x 4 (xTopLeft, yTopLeft, width, height).
    
    c_data = np.copy(data)
    
    # Remove face markers - they are intermittent.
    _, idxLBMarkers = getOpenPoseMarkerLowerBodyNames()
    idxToRemove = np.hstack([np.arange(i*3,i*3+3) for i in idxLBMarkers])
    c_data = np.delete(c_data, idxToRemove, axis=1)    
    
    # nan the data if below a threshold
    confData = c_data[:,2::3]<confidenceThreshold
    confMask = np.repeat(confData, 3, axis=1)
    c_data[confMask] = np.nan  
    nonNanRows = np.argwhere(np.any(~np.isnan(c_data), axis=1))
    
    bbox = np.zeros((c_data.shape[0], 4))
    bbox[nonNanRows,0] = np.nanmin(c_data[nonNanRows,0::3], axis=2)
    bbox[nonNanRows,1] = np.nanmin(c_data[nonNanRows,1::3], axis=2)
    bbox[nonNanRows,2] = (np.nanmax(c_data[nonNanRows,0::3], axis=2) - 
                          np.nanmin(c_data[nonNanRows,0::3], axis=2))
    bbox[nonNanRows,3] = (np.nanmax(c_data[nonNanRows,1::3], axis=2) - 
                          np.nanmin(c_data[nonNanRows,1::3], axis=2))
    
    # Go a bit above head (this is for image-based tracker).
    bbox[:,1] = np.maximum(0, bbox[:,1] - .05 * bbox[:,3])
    bbox[:,3] = bbox[:,3] * 1.05
    
    return bbox

#%%
def plot_multiple_markers(marker_data, marker_indices=None):
    """
    Plot x, y, z position over time for multiple markers.

    Parameters:
        marker_data: np.array of shape [nFrames, nMarkers, 3]
        marker_indices: list of marker indices to plot (defaults to all)
    """
    nFrames, nMarkers, _ = marker_data.shape
    time = np.arange(nFrames)

    if marker_indices is None:
        marker_indices = list(range(nMarkers))  # Plot all if none specified

    colors = plt.cm.get_cmap('tab10', 2)

    fig, ax = plt.subplots()

    for idx, m in enumerate(marker_indices):
        x = marker_data[:, m, 0]
        y = marker_data[:, m, 1]
        ax.plot(time, x, label=f'Marker {m}', color='red')
        ax.plot(time, y, label=f'Marker {m}', color='blue')

    ax.set_ylabel('Marker Position')


    fig.suptitle('Marker Trajectories Over Time')
    plt.tight_layout()
    plt.show()
    
def plot_multiple_markers_3D(marker_data, marker_names, marker_indices=None):
    """
    Plot x, y, z position over time for multiple markers.

    Parameters:
        marker_data: np.array of shape [nFrames, nMarkers, 3]
        marker_indices: list of marker indices to plot (defaults to all)
    """
    nFrames, nMarkers, _ = marker_data.shape
    time = np.arange(nFrames)

    if marker_indices is None:
        marker_indices = list(range(nMarkers))  # Plot all if none specified

    colors = plt.cm.get_cmap('tab10', 3)

    fig, ax = plt.subplots()

    for idx, m in enumerate(marker_indices):
        x = marker_data[:, m, 0]
        y = marker_data[:, m, 1]
        z = marker_data[:, m, 2]
        ax.plot(time, x, label=f'Marker {m}', color='red')
        ax.plot(time, y, label=f'Marker {m}', color='blue')
        ax.plot(time, z, label=f'Marker {m}', color='green')

    ax.set_ylabel('X Position')


    fig.suptitle(marker_names[marker_indices[0]] + ' Trajectories Over Time')
    plt.tight_layout()
    plt.show()
    
#%%
def pattern_based_fill(primary_kp, reference_kp, segment_length=None):
    """
    Fill missing values in `primary_kp` using the trajectory of `reference_kp`
    and an average offset (primary - reference) computed from valid frames.
    
    Arguments:
    - primary_kp: (n_frames, 3) array with NaNs in missing entries
    - reference_kp: (n_frames, 3) array with reference trajectory
    
    Returns:
    - filled: (n_frames, 3) array with missing values in primary_kp filled
    """
    filled = primary_kp.copy()

    # Identify where both primary and reference are valid
    valid_mask = ~np.isnan(primary_kp).any(axis=1) & ~np.isnan(reference_kp).any(axis=1)
    
    if not valid_mask.any():
        raise ValueError("No valid data to compute offset.")
    
    # Compute average offset
    offset = np.nanmean(primary_kp[valid_mask] - reference_kp[valid_mask], axis=0)

    # Fill where primary is NaN and reference is valid
    for i in range(len(primary_kp)):
        if np.isnan(primary_kp[i]).any() and not np.isnan(reference_kp[i]).any():
            filled[i] = reference_kp[i] + offset

    return filled

#%%
def fill_with_single_reference(target_marker, reference_marker,n=5):
    """
    Fill missing values in target_marker using average offset from n frames
    before and after the gap, based on a single reference marker.

    Args:
        target_marker: (T, 3) np.array with NaNs
        reference_marker: (T, 3) np.array with NaNs allowed
        n: number of frames before and after to use for computing local offset

    Returns:
        filled_target: (T, 3) np.array with missing values filled where possible
    """
    T = target_marker.shape[0]
    filled_target = target_marker.copy()

    for t in range(T):
        if np.isnan(target_marker[t]).any() and not np.isnan(reference_marker[t]).any():
            # Collect valid frames before and after
            before_idxs = []
            after_idxs = []

            # Search backward
            i = t - 1
            while i >= 0 and len(before_idxs) < n:
                if not np.isnan(target_marker[i]).any() and not np.isnan(reference_marker[i]).any():
                    before_idxs.append(i)
                i -= 1

            # Search forward
            i = t + 1
            while i < T and len(after_idxs) < n:
                if not np.isnan(target_marker[i]).any() and not np.isnan(reference_marker[i]).any():
                    after_idxs.append(i)
                i += 1

            use_idxs = before_idxs + after_idxs
            if len(use_idxs) >= 2:  # need at least 2 points to compute a good offset
                offsets = target_marker[use_idxs] - reference_marker[use_idxs]
                mean_offset = np.mean(offsets, axis=0)
                filled_target[t] = reference_marker[t] + mean_offset
            # else: leave it NaN

    return filled_target
# %%
def estimate_hip_joint_centers(shoulder_R, shoulder_L, height):
    """
    Estimate left and right hip joint centers from shoulder joint centers and subject height.
    Only estimates when shoulder_L and shoulder_R are both valid (not NaN).
    
    Args:
        shoulder_L: (N, 3) numpy array of left shoulder joint centers
        shoulder_R: (N, 3) numpy array of right shoulder joint centers
        height: float, subject's height in meters
    
    Returns:
        hip_L: (N, 3) numpy array of left hip joint centers (NaN if shoulders invalid)
        hip_R: (N, 3) numpy array of right hip joint centers (NaN if shoulders invalid)
    """
    # Initialize outputs with NaNs
    N = shoulder_L.shape[0]
    hip_L = np.full((N, 3), np.nan)
    hip_R = np.full((N, 3), np.nan)
    
    # Iterate over time points
    for i in range(N):
        sl = shoulder_L[i]
        sr = shoulder_R[i]
        
        if not np.any(np.isnan(sl)) and not np.any(np.isnan(sr)):
            shoulder_mid = (sl + sr) / 2
            pelvis_center = shoulder_mid.copy()
            pelvis_center[1] -= 0.22 * height  # drop vertically (Y-up assumption)

            # Direction from L to R shoulder
            right_vec = sr - sl
            norm = np.linalg.norm(right_vec)
            if norm > 0:
                right_vec /= norm
            else:
                continue  # skip frame if shoulder markers are too close
            
            hip_offset = 0.07 * height
            hip_R[i] = pelvis_center + hip_offset * right_vec
            hip_L[i] = pelvis_center - hip_offset * right_vec
    

    return hip_R, hip_L

#%%
def estimate_hips_from_shoulders(neck, Lshoulder, Rshoulder, height_m):
    """
    Estimate hip joint centers from upper body markers and height.

    Args:
        C7: (T, 3) array of neck base marker
        L_shoulder, R_shoulder: (T, 3) arrays for acromion markers
        height_m: child height in meters

    Returns:
        hip_L, hip_R: (T, 3) arrays for left and right hip joint centers
    """
    T = neck.shape[0]
    hip_L = np.full((T, 3), np.nan)
    hip_R = np.full((T, 3), np.nan)

    pelvis_width = 0.2 * height_m
    downward_offset = 0.25 * height_m  # distance from shoulders to hips

    for t in range(T):
        if any(np.isnan(arr[t]).any() for arr in [neck, Lshoulder, Rshoulder]):
            continue

        # Shoulder axis (left to right)
        shoulder_axis = Rshoulder[t] - Lshoulder[t]
        shoulder_axis /= np.linalg.norm(shoulder_axis)

        # Midpoint of shoulders
        shoulder_mid = (Lshoulder[t] + Rshoulder[t]) / 2

       # Global forward vector assumption (Y is forward)
        global_forward = np.array([0, 0, 1])  # Modify if needed
        
        # Estimate spine axis as cross product (perpendicular to shoulder & forward)
        spine_axis = np.cross(shoulder_axis, global_forward)
        spine_axis /= np.linalg.norm(spine_axis)
        
        # Flip if pointing upward
        if spine_axis[1] > 0:
            spine_axis = -spine_axis

        # Sagittal axis (approx front-back) = shoulder × spine
        sagittal_axis = np.cross(shoulder_axis, spine_axis)
        sagittal_axis /= np.linalg.norm(sagittal_axis)

        # Estimate pelvis center ~0.35 * height below shoulder midpoint
        pelvis_center = shoulder_mid + downward_offset * spine_axis

        # Estimate hip positions by offsetting pelvis center
        hip_R[t] = pelvis_center - (pelvis_width / 2)*shoulder_axis
        hip_L[t] = pelvis_center + (pelvis_width / 2)*shoulder_axis

    return hip_L, hip_R 
# %%
def main(subj, startTime, endTime, trialName,
         runMarkerAugmentation=True, scaleModel=True, 
         runInverseKinematics=True,fps=30):
    import sys
    sys.path.append('OpenCap')
    
    augmenterModel='v0.3'
    augmenterModelName = 'LSTM'
    offset = False
    
    #base path  
    projectDir = os.path.dirname(os.path.abspath(__file__))
    
    # meta data
    from utils import importMetadata
    from utilsOpenSim import runScaleTool, getScaleTimeRange, runIKTool
    sessionDir = os.path.join(projectDir, subj)
    sessionMetadata = importMetadata(os.path.join(sessionDir,
                                                          'sessionMetadata.yaml'))
    
    # opensim stuff
    openCapDir = os.path.join(projectDir,'OpenCap')
    openSimPipelineDir = os.path.join(openCapDir, 'openSimPipeline')
    
    # augmentor data
    augmenterDir = os.path.join(openCapDir, "MarkerAugmenter")
    output_folder = os.path.join(sessionDir, "MarkerData")
    trc_file_path = os.path.join(output_folder,trialName, "pose_keypoints.trc")
    pathAugmentedOutputFiles = os.path.join(
            output_folder, trialName, "keypoints_augmented.trc")
    
    # Camera indices (update these for your setup)
    path_main = os.path.join(sessionDir,"Videos")
    cameras = ['Cam1','Cam2']
    
    # Intrinsics/extrinsics calculation
    import extrinsicsIntrinsicsCalc as exInt
    exInt.main(subj,path=projectDir,camList=cameras,
               metaData=sessionMetadata['checkerBoard'])  
    
    # Proceed to the next steps
    camera_matrices = []
    dist_coeffs = []
    extrinsics = []
    for i, cam in enumerate(cameras):
        path_intrinsics = os.path.join(path_main,cam,'CameraParameters','cameraIntrinsicsExtrinsics.pickle')
        parameters = read_pickle_file(path_intrinsics)
        R = parameters['rotation']
        T = parameters['translation']
        ex = np.hstack((R, T.reshape(-1, 1)))
        
        camera_matrices.append(parameters['intrinsicMat'])
        dist_coeffs.append(parameters['distortion'])
        extrinsics.append(ex)
             
    # Initialize pose triangulator
    triangulator = PoseTriangulator(camera_matrices, dist_coeffs, extrinsics)

    # Check if pickle file exists
    pickle_file = trialName + '_all_keypoints_3d.pickle'
    if os.path.exists(os.path.join(sessionDir,pickle_file)):
        all_keypoints_3d = read_pickle_file(os.path.join(sessionDir,pickle_file))

# %%
    else:
        # Initialize video capture for both cameras (replace with your video file paths or webcam index)
        captures = []
        videoPaths = []
        for i, cam in enumerate(cameras):
            video_name = trialName
            file_name = video_name + '_cut.mp4'
            path_video_cut = os.path.join(path_main,cam,trialName,file_name)
            
            #shorten the video
            if not os.path.exists(path_video_cut):
                file_name = video_name + '.mp4'
                path_video = os.path.join(path_main,cam,trialName,file_name)
                if startTime is None:
                    startTime = 0
                if endTime is None:
                    cap = cv2.VideoCapture(path_video)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    endTime = frame_count / fps
                
                cut_video(path_video, path_video_cut, startTime, endTime)
              
            #correct the video
            file_name = video_name + '_corrected.mp4'
            path_video_corrected = os.path.join(path_main,cam,trialName,file_name)
            if not os.path.exists(path_video_corrected):          
                #correct the gopro video
                command = [
                    "ffmpeg",
                    "-i", path_video_cut,
                    "-preset", "veryfast",
                    "-crf", "23",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    path_video_corrected
                ]
                subprocess.run(command)
                
            videoPaths.append(path_video_corrected)

        # Video synchronization
        file_name = 'synced_' + video_name + '_corrected.mp4'
        path_video_synced = os.path.join(path_main,cam,trialName,file_name)
        if not os.path.exists(path_video_synced):
            from videoSynch import VideoSynchronizer
            synchronizer = VideoSynchronizer(videoPaths)
            offsets = synchronizer.run_synchronization(visualize=True)
            
            print("\nSummary of time offsets (in seconds) relative to reference video:")
            for i, offset in enumerate(offsets):
                if i == synchronizer.reference_index:
                    print(f"Video {i}: {offset:.3f}s (REFERENCE)")
                else:
                    print(f"Video {i}: {offset:.3f}s (confidence: {synchronizer.confidence_scores[i]:.2f})")
            
        # Detect keypoints
        from poseDetection import runOpenPoseCMD
        openpose_json_dirs = [os.path.join(path_main, cam, trialName, 'openpose_json') for cam in cameras]
        
        # Run OpenPose if keypoints are not already extracted
        for cam, json_dir in zip(cameras, openpose_json_dirs):
            print("Starting Pose Detection for " + cam)
            if not os.path.exists(json_dir):
                os.makedirs(json_dir, exist_ok=True)
                video_name = 'synced_' + trialName + "_corrected.mp4"
                path_video = os.path.join(path_main, cam, trialName, video_name)
                runOpenPoseCMD(
                    pathOpenPose='C:/openpose/',
                    resolutionPoseDetection='default',
                    cameraDirectory=os.path.join(path_main,cam,trialName),
                    fileName=video_name,
                    openposeJsonDir=json_dir,
                    pathOutputVideo='',  # optional if not saving visualized video
                    trialPrefix='',
                    generateVideo=True,
                    videoFullPath=path_video,
                    pathOutputJsons=json_dir
                )
    
        frame_count = 0
        
        # Load keypoints from JSON files
        if not os.path.exists(os.path.join(sessionDir,trialName + 'all_keypoints_3d.pickle')):
            keypoints_list = {}
            for index, json_dir in enumerate(openpose_json_dirs):
                keypoints_2d = load_openpose_keypoints(json_dir)
                if keypoints_2d is None:
                    print(f"Missing or empty keypoints for frame {frame_count}")
                    break
                keypoints_list[f"Cam{index+1}"] = keypoints_2d
                
            # Find the correct person to track across the frames
            keypointsL = []
            for index,cam in enumerate(cameras):
                file_name = 'synced_' + video_name + '_corrected.mp4'
                videoPath = os.path.join(path_main,cam,trialName,file_name)
                
                allPeople = keypoints_list[cam]
                
                if len(allPeople) >1:    #this looks at the number of detected people in the first frame
                    nFrames = len(allPeople[0])
                    print(f"Found {len(allPeople)} people with {nFrames} frames of data")
                    personToTrack, startFrame = frame_navigation(videoPath, allPeople, nFrames)
                    
                    confidenceThresholdForBB=0.4
                    bbFromKeypoints = [keypointsToBoundingBox(data,confidenceThreshold=confidenceThresholdForBB) for data in allPeople]
                    
                    if personToTrack is not None:
                        startPerson = personToTrack
                        startBb = bbFromKeypoints[startPerson][startFrame]
                    else:
                        # If no person is found
                        print("No person selected. Returning zeroed arrays.")
                        key2D = np.zeros((25, nFrames, 2))
                        confidence = np.zeros((25, nFrames))
                        return key2D, confidence
            
                    # initialize output data
                    res = np.zeros((nFrames, 75))
                    # res[:] = np.nan
                      
                    poseDetector = "OpenPose"
                    res = trackKeypointBox(videoPath , startBb , allPeople ,
                                            bbFromKeypoints , res , cam, frameStart = startFrame, 
                                            frameIncrement = -1 , visualize=False, 
                                            poseDetector=poseDetector)
                    
                    # track this bounding box forward until it can't be tracked            
                    res = trackKeypointBox(videoPath , startBb , allPeople , 
                                            bbFromKeypoints , res, cam, frameStart = startFrame, 
                                            frameIncrement = 1 , visualize=False, 
                                            poseDetector=poseDetector)
                else:
                    res = allPeople[0]
                
                key2D = np.zeros((25,nFrames,3))
                for i in range(0,25):
                    key2D[i,:,0:2] = res[:,i*3:i*3+2]
                    key2D[i,:,2] = res[:,i*3+2]
                    
                #convert 0s to NaNs
                key2D[key2D == 0] = np.nan
                                
                keypointsL.append(key2D)
                
            # remove unnecessary keypoints and shape it in the correct form
            selected_indices = [0,1,2,3,4,5,6,7,15,16]
            selected_indices = np.array(selected_indices)
            for i, cam in enumerate(keypointsL):
                keypointsL[i] = keypointsL[i].transpose([1,0,2])
                keypointsL[i] = keypointsL[i][:,selected_indices,:]
                
            marker_names = [triangulator.landmark_names[i] for i in selected_indices]
                    
            all_keypoints_3d = []    
            for frame in range(nFrames):   
                timestamp = frame  
                # Triangulate keypoints
                keypoints_3d = triangulator.triangulate_pose(keypointsL, timestamp)
                
                # Store the 3D keypoints
                all_keypoints_3d.append(keypoints_3d)                          
                
            print(f"\nCollected {len(all_keypoints_3d)} frames of keypoint data")
            
            all_keypoints_3d = np.array(all_keypoints_3d)
            
            #save it as a pickle file
            pickleName = trialName + 'all_keypoints_3d.pickle'
            file_pickle = os.path.join(sessionDir,pickleName)
            savePickle(file_pickle,all_keypoints_3d)
    # %%
    #check if .trc exists already
    if os.path.exists(trc_file_path):
        pass
    else:
        selected_indices = [0,1,2,3,4,5,6,7,15,16]
        marker_names = [triangulator.landmark_names[i] for i in selected_indices]
        
        all_keypoints_3d = np.array(all_keypoints_3d)
        
                
        # Visualize things with GUI
        # import tkinter as tk
        # from GUI import TrajectoryViewer
        
        # root = tk.Tk()
        # app = TrajectoryViewer(root, all_keypoints_3d, marker_names)
        # root.mainloop()  
        
        # Filter by velocity
        _,kP,_ = all_keypoints_3d.shape
        for i in range(kP):
            all_keypoints_3d[:,i,:] = filter_markers_by_velocity(all_keypoints_3d[:,i,:])
        
        # # Interpolate
        interpolated_keypoints = []        
        # For each keypoint (e.g., nose, left shoulder, etc.)
        for kp_idx in range(all_keypoints_3d.shape[1]):
            # Extract this keypoint's trajectory across all frames
            keypoint_trajectory = all_keypoints_3d[:, kp_idx, :]
            
            # Interpolate this keypoint's trajectory
            interpolated_trajectory = interpolate(
                keypoint_trajectory,
                method='spline', 
                max_gap_frames=30
            )         
            interpolated_keypoints.append(interpolated_trajectory)
            
        #Transpose to get back to original format [frame][keypoint][xyz]
        all_keypoints_3d = np.array(interpolated_keypoints).transpose(1, 0, 2)
        
        # Filter by velocity #2
        _,kP,_ = all_keypoints_3d.shape
        for i in range(kP):
            all_keypoints_3d[:,i,:] = filter_markers_by_velocity(all_keypoints_3d[:,i,:])
        
        # # Interpolate
        interpolated_keypoints = []        
        # For each keypoint (e.g., nose, left shoulder, etc.)
        for kp_idx in range(all_keypoints_3d.shape[1]):
            # Extract this keypoint's trajectory across all frames
            keypoint_trajectory = all_keypoints_3d[:, kp_idx, :]
            
            # Interpolate this keypoint's trajectory
            interpolated_trajectory = interpolate(
                keypoint_trajectory,
                method='spline', 
                max_gap_frames=30
            )         
            interpolated_keypoints.append(interpolated_trajectory)
            
        #Transpose to get back to original format [frame][keypoint][xyz]
        all_keypoints_3d = np.array(interpolated_keypoints).transpose(1, 0, 2)
        
        # Filter by velocity #3
        _,kP,_ = all_keypoints_3d.shape
        for i in range(kP):
            all_keypoints_3d[:,i,:] = filter_markers_by_velocity(all_keypoints_3d[:,i,:])
        
        # # Interpolate
        interpolated_keypoints = []        
        # For each keypoint (e.g., nose, left shoulder, etc.)
        for kp_idx in range(all_keypoints_3d.shape[1]):
            # Extract this keypoint's trajectory across all frames
            keypoint_trajectory = all_keypoints_3d[:, kp_idx, :]
            
            # Interpolate this keypoint's trajectory
            interpolated_trajectory = interpolate(
                keypoint_trajectory,
                method='spline', 
                max_gap_frames=30
            )         
            interpolated_keypoints.append(interpolated_trajectory)
            
        #Transpose to get back to original format [frame][keypoint][xyz]
        all_keypoints_3d = np.array(interpolated_keypoints).transpose(1, 0, 2)
        
        
        '''
        Nose - 0
        Neck - 1
        RShoulder - 2
        RElbow - 3
        RWrist - 4
        LShoulder - 5
        LElbow - 6
        LWrist - 7
        REye - 8
        LEye - 9
        '''
        
        
        # Try pattern-based filling
        # Nose to REye
        primary_kp = all_keypoints_3d[:,8,:]
        reference_kp = all_keypoints_3d[:,0,:]
        all_keypoints_3d[:,8,:] = fill_with_single_reference(primary_kp, reference_kp)
        #Nose to LEye
        primary_kp = all_keypoints_3d[:,9,:]
        reference_kp = all_keypoints_3d[:,0,:]
        all_keypoints_3d[:,9,:] = fill_with_single_reference(primary_kp, reference_kp)
        #Neck to RShoulder
        primary_kp = all_keypoints_3d[:,2,:]
        reference_kp = all_keypoints_3d[:,1,:]
        all_keypoints_3d[:,2,:] = fill_with_single_reference(primary_kp, reference_kp)
        #Neck to LShoulder
        primary_kp = all_keypoints_3d[:,5,:]
        reference_kp = all_keypoints_3d[:,1,:]
        all_keypoints_3d[:,5,:] = fill_with_single_reference(primary_kp, reference_kp)
        #RShoulder to RElbow
        primary_kp = all_keypoints_3d[:,3,:]
        reference_kp = all_keypoints_3d[:,2,:]
        all_keypoints_3d[:,3,:] = fill_with_single_reference(primary_kp, reference_kp)
        #LShoulder to LElbow
        primary_kp = all_keypoints_3d[:,6,:]
        reference_kp = all_keypoints_3d[:,5,:]
        all_keypoints_3d[:,6,:] = fill_with_single_reference(primary_kp, reference_kp)
        #RElbow to RWrist
        primary_kp = all_keypoints_3d[:,4,:]
        reference_kp = all_keypoints_3d[:,3,:]
        all_keypoints_3d[:,4,:] = fill_with_single_reference(primary_kp, reference_kp)
        #LElbow to LWrist
        primary_kp = all_keypoints_3d[:,7,:]
        reference_kp = all_keypoints_3d[:,6,:]
        all_keypoints_3d[:,7,:] = fill_with_single_reference(primary_kp, reference_kp)
        
        # low-pass
        all_keypoints_3d = low_pass_filter(all_keypoints_3d)
        
        
        # MEDIAN FILTER
        from scipy.signal import medfilt
        for mrk in range(len(marker_names)):
            signal = all_keypoints_3d[:,mrk,:]
            all_keypoints_3d[:,mrk,:] = np.apply_along_axis(medfilt, 0, signal, kernel_size=5) 
            plot_multiple_markers_3D(all_keypoints_3d,marker_names, marker_indices=[mrk])
        
        
        #estimate hip positions
        height = sessionMetadata['height_m']
        RHip,LHip = estimate_hip_joint_centers(all_keypoints_3d[:,5,:],
                                               all_keypoints_3d[:,2,:],
                                               height*1000)
        
        RHip = RHip[:, np.newaxis, :]
        LHip = LHip[:, np.newaxis, :]
        all_keypoints_3d = np.concatenate((all_keypoints_3d,RHip),axis=1)
        all_keypoints_3d = np.concatenate((all_keypoints_3d,LHip),axis=1)
        
        marker_names.append('RHip')
        marker_names.append('LHip')    
        
        # Find the longest complete marker sequence
        (start_frame, end_frame), sequence_length = find_longest_complete_sequence(all_keypoints_3d)
        complete_sequence = all_keypoints_3d[start_frame:end_frame+1,:,:]
        
        
        # Create TRC file        
        if not os.path.isdir(os.path.join(output_folder,trialName)):
            os.mkdir(os.path.join(output_folder,trialName))
        trc_file_path = os.path.join(output_folder, trialName, "pose_keypoints.trc")
        trc_file = open(trc_file_path, "w")
        write_trc_header(trc_file, fps, marker_names)
        
        # Write interpolated keypoints to TRC file
        for frame_idx, frame_data in enumerate(complete_sequence):
            timestamp = frame_idx / fps
            write_trc_frame(trc_file, frame_idx + 1, timestamp, frame_data)
        
        # Close TRC file and update header
        trc_file.close()
        update_trc_header(trc_file_path, len(complete_sequence))
        
        print(f"Output saved to {output_folder}")
        print(f"TRC file saved to {trc_file_path}")
    
    if runMarkerAugmentation:
        if not os.path.exists(pathAugmentedOutputFiles):
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            import traceback
            from utilsAugmenter import augmentTRC
    
            print('Augmenting marker set')
            try:
                vertical_offset = augmentTRC(
                    trc_file_path,sessionMetadata['mass_kg'], 
                    sessionMetadata['height_m'], pathAugmentedOutputFiles,
                    augmenterDir, augmenterModelName=augmenterModelName,
                    augmenter_model=augmenterModel, offset=offset)
            except Exception as e:
                if len(e.args) == 2: # specific exception
                    raise Exception(e.args[0], e.args[1])
                elif len(e.args) == 1: # generic exception
                    exception = "Marker augmentation failed."
                    raise Exception(exception, traceback.format_exc())
                               
    if scaleModel and not os.path.exists(os.path.join(
            sessionDir, 'Model','MASI-upperOnly-markers_scaled.osim')):      
        # Get time range.
        thresholdPosition = 0.003
        maxThreshold = 0.015
        increment = 0.001
        success = False
        while thresholdPosition <= maxThreshold and not success:
            try:
                timeRange4Scaling = getScaleTimeRange(
                    pathAugmentedOutputFiles,
                    thresholdPosition=thresholdPosition,
                    thresholdTime=0.1, removeRoot=True)
                success = True
            except Exception as e:
                print(f"Attempt identifying scaling time range with thresholdPosition {thresholdPosition} failed: {e}")
                thresholdPosition += increment  # Increase the threshold for the next iteration
        
        subjectMass = sessionMetadata['mass_kg']
    
        pathGenericSetupFile = os.path.join(openSimPipelineDir,
                                            'Setup_scaling_MASI_noEars.xml')
        pathGenericModel = os.path.join(openSimPipelineDir, 
                                        'MASI-upperOnly-markers-slider_noEars.osim')
        suffix_model = 'upperBody_' + subj
        outputScaledModelDir = os.path.join(sessionDir, 'Model', trialName)
        scaledModel = runScaleTool(pathGenericSetupFile, pathGenericModel, subjectMass,
                         pathAugmentedOutputFiles, timeRange4Scaling, outputScaledModelDir, 
                         subjectHeight=sessionMetadata['height_m'], 
                         suffix_model=suffix_model)
        
        print("Scaling completed!")
        
    if runInverseKinematics:
        #do this and do that
        outputIKDir = os.path.join(sessionDir, 'Kinematics', trialName)
        os.makedirs(outputIKDir, exist_ok=True)
        # Check if there is a scaled model.
        pathScaledModel = os.path.join(sessionDir, 'Model', trialName,
                                        sessionMetadata['openSimModel'] + 
                                        "_scaled.osim")
        if os.path.exists(pathScaledModel):
            # Path setup file.
            genericSetupFile4IKName = 'Setup_IK_modified-AKC02.xml'
            pathGenericSetupFile4IK = os.path.join(
                openSimPipelineDir, 'IK', genericSetupFile4IKName)
            # Path TRC file.
            pathTRCFile4IK = pathAugmentedOutputFiles
            # Run IK tool. 
            try:
                pathOutputIK, pathModelIK = runIKTool(
                    pathGenericSetupFile4IK, pathScaledModel, 
                    pathTRCFile4IK, outputIKDir)
            except Exception as e:
                if len(e.args) == 2: # specific exception
                    raise Exception(e.args[0], e.args[1])
                elif len(e.args) == 1: # generic exception
                    exception = "Inverse kinematics failed. Verify your setup and try again. Visit https://www.opencap.ai/best-pratices to learn more about data collection and https://www.opencap.ai/troubleshooting for potential causes for a failed trial."
                    raise Exception(exception, traceback.format_exc())
        else:
            raise ValueError("No scaled model available.")       
            
        

if __name__ == "__main__":
    main()

