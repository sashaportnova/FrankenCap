# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 11:24:23 2025

@author: Sasha Portnova

This is a code for GUI (2D): creates a separate window for each camera with its
own functionalities
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from main import filter_markers_by_velocity, low_pass_filter, interpolate,pattern_based_fill

class TrajectoryViewer:
    def __init__(self, root, keypoint_data, marker_names, cam_index):
        self.root = root
        self.root.title(f"Camera {cam_index + 1} - Keypoint Trajectory Viewer")
        self.keypoint_data = keypoint_data
        self.current_kp = tk.IntVar(value=0)
        self.marker_names = marker_names
        self.create_widgets()
        self.plot_trajectory()

    def create_widgets(self):
        kp_options = [f"[{name} marker]" for name in self.marker_names]
        self.dropdown = ttk.Combobox(self.root, values=kp_options, state="readonly")
        self.dropdown.current(0)
        self.dropdown.pack()
        self.dropdown.bind("<<ComboboxSelected>>", self.update_plot)

        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()

        # Filtering Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)
        tk.Label(button_frame, text="Applied to all markers:").pack(side='left')
        tk.Button(button_frame, text='Low Pass Filter', command=self.apply_lowpass_filter).pack(side='left')

        velocity_frame = tk.Frame(self.root)
        velocity_frame.pack(pady=5)
        tk.Label(velocity_frame, text="Applied to a single marker:").pack(side='left')
        tk.Button(velocity_frame, text="Filter by Velocity", command=self.apply_filterVelocity).pack(side='left')

        # Interpolation Controls
        interp_frame = tk.Frame(self.root)
        interp_frame.pack(pady=5)
        tk.Label(interp_frame, text="Max frame gap:").pack(side='left')
        self.frame_gap_entry = tk.Entry(interp_frame, width=5)
        self.frame_gap_entry.insert(0, "10")
        self.frame_gap_entry.pack(side='left', padx=5)
        tk.Button(interp_frame, text="Interpolate Keypoint", command=self.run_interpolation).pack(side='left')

        # Pattern Fill Controls
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
        tk.Button(pattern_frame, text="Pattern Fill", command=self.run_pattern_fill).pack(side='left')

    def update_plot(self, event=None):
        self.current_kp.set(self.dropdown.current())
        self.plot_trajectory()

    def plot_trajectory(self):
        self.ax.clear()
        kp_idx = self.current_kp.get()
        traj = self.keypoint_data[:, kp_idx, :]
        self.ax.plot(traj[:, 0], label='X')
        self.ax.plot(traj[:, 1], label='Y')
        self.ax.set_title(f"Trajectory for {self.marker_names[kp_idx]} marker")
        self.ax.legend()
        self.canvas.draw()

    def apply_filterVelocity(self):
        idx = self.current_kp.get()
        self.keypoint_data[:, idx, :] = filter_markers_by_velocity(self.keypoint_data[:, idx, :])
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
        idx = self.current_kp.get()
        self.keypoint_data[:, idx, :] = interpolate(self.keypoint_data[:, idx, :], method='spline', max_gap_frames=max_gap)
        self.plot_trajectory()

    def run_pattern_fill(self):
        primary_idx = self.primary_dropdown.current()
        reference_idx = self.reference_dropdown.current()
        if primary_idx == reference_idx:
            print("Primary and reference keypoints must be different.")
            return
        filled = pattern_based_fill(
            self.keypoint_data[:, primary_idx, :],
            self.keypoint_data[:, reference_idx, :]
        )
        self.keypoint_data[:, primary_idx, :] = filled
        if self.current_kp.get() == primary_idx:
            self.plot_trajectory()

# Sample usage
if __name__ == "__main__":
    dummy_data = [np.random.rand(100, 5, 2) for _ in range(2)]
    dummy_markers = [f"Marker{i}" for i in range(5)]
    for i, cam_data in enumerate(dummy_data):
        window = tk.Tk()
        TrajectoryViewer(window, cam_data, dummy_markers, cam_index=i)
    tk.mainloop()