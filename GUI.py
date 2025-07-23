# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 10:56:15 2025

@author: Sasha Portnova

This is a code for the GUI. We will start by plotting keypoint trajectories,
filter the plot by keypoints, and have simple buttons to apply filters
and update the plot.
"""
import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.signal import medfilt

from main import filter_markers_by_velocity, low_pass_filter, interpolate,pattern_based_fill

class TrajectoryViewer:
    def __init__(self, root, keypoint_data, marker_names):
        self.root = root
        self.root.title("Keypoint Trajectory Viewer")
        self.keypoint_data = keypoint_data  # shape: (n_frames, n_keypoints, 3)
        self.current_kp = tk.IntVar(value=0)
        self.marker_names = marker_names

        self.create_widgets()
        self.plot_trajectory()

    def create_widgets(self):
        # Dropdown for selecting keypoints
        markerNames = self.marker_names
        kp_options = [f"[{markerNames[i]} marker]" for i in range(self.keypoint_data.shape[1])]
        self.dropdown = ttk.Combobox(self.root, values=kp_options, state="readonly")
        self.dropdown.current(0)
        self.dropdown.pack()
        self.dropdown.bind("<<ComboboxSelected>>", self.update_plot)

        # Plot figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        
        # Add buttons
        # --- Frame for Button Section ---
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)
        
        #Label next to the buttons
        label = tk.Label(button_frame, text="Applied to all markers:")
        label.pack(side='left', padx=(0, 5))
        
        self.btn_filtLowPass = tk.Button(button_frame, text='Low Pass Filter', command=self.apply_lowpass_filter)
        self.btn_filtLowPass.pack(side='left')
        
        # --- Frame for Velocity Filter Section ---
        velocity_frame = tk.Frame(self.root)
        velocity_frame.pack(pady=5)
        
        #Label next to the buttons
        label = tk.Label(velocity_frame, text="Applied to a single marker:")
        label.pack(side='left', padx=(0, 5))
        self.btn_filtVelocity = tk.Button(velocity_frame, text="Filter by Velocity", command=self.apply_filterVelocity)
        self.btn_filtVelocity.pack(side='left')
        
        # --- Frame for Interpolation Section ---
        interp_frame = tk.Frame(self.root)
        interp_frame.pack(pady=5)
    
        # Label next to entry field
        label = tk.Label(interp_frame, text="Max frame gap:")
        label.pack(side='left', padx=(0, 5))
    
        # Input field for frame_gap
        self.frame_gap_entry = tk.Entry(interp_frame, width=5)
        self.frame_gap_entry.insert(0, "10")  # default value
        self.frame_gap_entry.pack(side='left', padx=(0, 10))
    
        # Interpolate button next to it
        self.interpolate_button = tk.Button(interp_frame, text="Interpolate Keypoint", command=self.run_interpolation)
        self.interpolate_button.pack(side='left')
        
        # --- Frame for pattern-based fill --- 
        pattern_frame = tk.Frame(self.root)
        pattern_frame.pack(pady=5)
    
        tk.Label(pattern_frame, text="Primary KP:").pack(side='left')
        self.primary_dropdown = ttk.Combobox(pattern_frame, values=kp_options, state="readonly", width=10)
        self.primary_dropdown.current(0)
        self.primary_dropdown.pack(side='left', padx=2)
    
        tk.Label(pattern_frame, text="Reference KP:").pack(side='left')
        self.reference_dropdown = ttk.Combobox(pattern_frame, values=kp_options, state="readonly", width=10)
        self.reference_dropdown.current(1)
        self.reference_dropdown.pack(side='left', padx=2)
    
        self.pattern_button = tk.Button(pattern_frame, text="Pattern Fill", command=self.run_pattern_fill)
        self.pattern_button.pack(side='left', padx=5)
        
        # --- Frame for median filter --- 
        median_frame = tk.Frame(self.root)
        median_frame.pack(pady=5)
        
        label = tk.Label(median_frame, text="Applied to a single marker:")
        label.pack(side='left', padx=(0, 5))
        self.btn_filtMedian = tk.Button(median_frame, text="Median Filter", command=self.apply_MedianFilter)
        self.btn_filtMedian.pack(side='left')

    def update_plot(self, event=None):
        self.current_kp.set(self.dropdown.current())
        self.plot_trajectory()

    def plot_trajectory(self):
        self.ax.clear()
        markerNames = self.marker_names
        kp_idx = self.current_kp.get()
        traj = self.keypoint_data[:, kp_idx, :]  # shape: (n_frames, 3)
        _,nDim = traj.shape
        
        if nDim == 3:
            self.ax.plot(traj[:, 0], label='X')
            self.ax.plot(traj[:, 1], label='Y')
            self.ax.plot(traj[:, 2], label='Z')
        elif nDim ==2:
            self.ax.plot(traj[:, 0], label='X')
            self.ax.plot(traj[:, 1], label='Y')
        self.ax.set_title(f"Trajectory for {markerNames[kp_idx]} marker")
        self.ax.legend()
        self.canvas.draw()
        
    def apply_filterVelocity(self):
        idx = self.current_kp.get()
        self.keypoint_data[:,idx,:] = filter_markers_by_velocity(self.keypoint_data[:,idx,:])
        self.plot_trajectory()
        
    def apply_MedianFilter(self):
        idx = self.current_kp.get()
        self.keypoint_data[:,idx,:] = np.apply_along_axis(medfilt, 0, self.keypoint_data[:,idx,:], kernel_size=5) 
        self.plot_trajectory()
        
    def apply_lowpass_filter(self):
        self.keypoint_data = low_pass_filter(self.keypoint_data)
        self.plot_trajectory()
        
    def run_interpolation(self):
        try:
            max_gap = int(self.frame_gap_entry.get())
        except ValueError:
            print("Invalid frame gap value.")
            return

        # Extract the selected keypoint's trajectory: shape (n_frames, 3)
        idx = self.current_kp.get()
        kp_traj = self.keypoint_data[:, idx, :]

        # Run your interpolation function (replace with your own)
        self.keypoint_data[:, idx, :] = interpolate(kp_traj, method='spline', max_gap_frames=max_gap)

        # Update keypoints and re-plot
        self.plot_trajectory()
        
    def run_pattern_fill(self):   
        primary_idx = self.primary_dropdown.current()
        reference_idx = self.reference_dropdown.current()
    
        if primary_idx == reference_idx:
            print("Primary and reference keypoints must be different.")
            return
    
        primary_kp = self.keypoint_data[:, primary_idx, :]
        reference_kp = self.keypoint_data[:, reference_idx, :]
    
        filled_kp = pattern_based_fill(primary_kp, reference_kp)
        self.keypoint_data[:, primary_idx, :] = filled_kp
    
        if self.current_kp.get() == primary_idx:
            print('Woohoo!')
            self.plot_trajectory()