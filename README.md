# FrankenCap
**FrankenCap** is an OpenCap-based pipeline for markerless tracking of upper-body and head kinematics in children using mobility devices, designed to handle lower-limb occlusion and seated postures. Developed at the [Neuromechanics and Mobility Lab](https://steelelab.me.uw.edu/) at the [University of Washington](https://www.washington.edu/).

This tool builds on the excellent foundation provided by the [OpenCap](https://github.com/stanfordnmbl/opencap-core) team at Stanford University. We are tremendously thankful for their beautifully structured, clean, and easy-to-follow code to make FrankenCap a relatity. The project name comes from the fact that they amount of times that we "surgically" took and put back together the original OpenCap code was par to that of the Frankenstein creation, hence the name.

---

## 🔍 Overview

FrankenCap enables 3D tracking of **head, neck, and upper-limb kinematics** in challenging environments, such as young children in early mobility devices like:

- [**GoBabyGo cars**](https://www.instructables.com/GoBabyGo-Make-a-Joystick-controlled-Ride-on-Car/)
- [**Permobil Explorer Mini**](https://www.permobil.com/en-us/products/power-wheelchairs/permobil-explorer-mini)

These systems occlude the lower body, preventing accurate whole-body tracking with most markerless systems. FrankenCap addresses this gap with:

- **Upper-body focused pose estimation**
- **Tracking in case of multi-person presence**
- **Filtering and smoothing to handle random occlusions that happen in chaotic data collection environments with pediatric users**

If you're studying **how children move and interact** in early mobility contexts and need to **track kinematics over time**, we hope this tool is useful and something you can build upon.

---

## ✅ Features

- 🎯 Upper-body kinematic tracking using multi-view OpenCap  
- 🔊 Audio-based synchronization for multiple camera videos  
- 👥 Supports tracking the person of interest in complex scenes with multiple people present
- 🧹 Robust to short-term occlusion with integrated filtering

---

## 🛠️ Installation

We recommend using a virtual environment:

```bash
python -m venv FrankenCap
source FrankenCap/bin/activate  # or FrankenCap\Scripts\activate on Windows
```

# Install dependencies:

1. Install OpenCap-core following their [instructions](https://github.com/stanfordnmbl/opencap-core) under the Installation tab.

2. Install librosa for audio-based synchronization:

```bash
pip install librosa
```

---

## 📸 Setup Instructions

### 🎥 Camera Configuration

- Use **two front-facing cameras** placed at converging angles to capture clear views of the child’s **head, arms, and upper torso** by two cameras simultaneously. If needed, you can try to use the superwide or wide angle lenses or options on your cameras.
<img width="974" height="548" alt="image" src="https://github.com/user-attachments/assets/83cacab6-3b59-482f-9f45-a07babe90882" />
- We recommend **GoPro cameras** mounted securely to the mobility device. Ensure cameras **remain fixed** throughout data collection.
- Recommended mounting accessories will be linked here soon.

### 🎯 Calibration

- Use a **4×5 35mm checkerboard** for camera calibration. [Link](https://calib.io/)
- If you use a different checkerboard pattern, update the relevant settings in the `sessionMetadata.yaml` file located in the subject/session folder.
- For *intrinsics* calibration: you will need to record a video of the calibration checkerboard floated in the frames between both cameras at different angles. Save these files as *intrinsics.mp4*.
- For *extrinsics* calibration: you will need to record a brief video of the checkerboard placed perpendicular to the cameras in the following setup (PHOTO INCLUDED BELOW). Save these files as *extrinsics.mp4*.
  <img width="759" height="630" alt="image" src="https://github.com/user-attachments/assets/a5769697-bc4f-4c1a-927d-91e187a2fb93" />
  
---

## 🗂️ Data Organization

Organize your data in the following folder structure:

```
FrankenCap/
└── P1/ # Subject folder (e.g., Participant 1)
├── sessionMetadata.yaml # Metadata file describing session setup
└── Videos/
├── Cam1/
│ ├── CameraParameters/
│ │ ├── intrinsics.mp4
│ │ └── extrinsics.mp4
│ ├── play/
|    └── play.mp4 # Trial video for Cam1
└── Cam2/
├── CameraParameters/
│ ├── intrinsics.mp4
│ └── extrinsics.mp4
│ ├── play/
|    └── play.mp4 # Trial video for Cam1
```

> ✅ Ensure the `sessionMetadata.yaml` file matches your setup:
> - Checkerboard dimensions and size
> - Height and mass of the participant (used for OpenSim scaling)
> - openSimModel is set to *MASI-upperOnly-markers-slider_noEars*

---

## 🎥 Video Recording Guidelines

While FrankenCap includes a relatively robust **audio-based synchronization algorithm**, we still recommend starting both cameras **as simultaneously as possible**. With our pipeline, we were able to handle up to ~5 seconds of misalignment between videos, but cleaner synchronization improves accuracy.

**Tips for video capture:**
- Try to **reduce background activity** at the beginning of the recording. Avoid having additional people in the frame behind or near the child from the time you start recording your activity of interest.
- This improves the accuracy of person detection and helps the system more reliably identify the **correct person of interest**.

---

## ▶️ Running the Pipeline

To begin analysis, open the `runAnalysis.py` script. This is the main entry point for processing your video data.

### 🔧 Configuration Steps

1. Open `runAnalysis.py` in your code editor.
2. Modify the following variables near the top of the script:

```python
subj = 'P1'           # Folder name for your participant (e.g., 'P1')
fps = 30              # Frames per second of your video (default is 30Hz)
startTime = None      # (Optional) Start time in seconds if you want to trim the video
endTime = None        # (Optional) End time in seconds
trialName = 'play'    # Name of the trial video (e.g., 'play.mp4')
```

---

## 👤 Selecting the Person of Interest

After running pose detection on both videos, if **multiple people are present** in the frame, the system will prompt you to select the correct subject.

You will see a message like: `Is the person of interest present in this frame? [y/n]`

- If you respond with **`y`**, a figure will be shown with each detected person labeled (`0`, `1`, `2`, etc.).
  - Select the correct person by typing the **index number** corresponding to the correct bounding box.
- If the keypoints are incorrect or not associated with the correct subject (e.g., the wrist is tracked on someone else's hip), type **`n`** to skip to the next frame.
  - The system will skip ahead by 10 frames and try again.

This selection process occurs **independently for each camera**.  
Once selected, the algorithm will automatically track the correct subject **throughout the full video**, both forward and backward, and triangulate their 3D positions.

---

## 🔄 FrankenCap Pipeline Steps

### 1️⃣ Video Processing
- Optional: **Trim** the video
- Optional: **Re-encode** to improve readability
- Perform **audio-based synchronization** to align multiple camera videos

### 2️⃣ Pose Detection
- Run **OpenPose** on each video to extract 2D keypoints
- Select the **person of interest** using the bounding-box selection prompt

### 3️⃣ Triangulation
- Convert 2D keypoints from multiple views into **3D coordinates**
- Save the 3D keypoints as a `.pickle` file

### 4️⃣ 3D Filtering
- **Velocity-based outlier removal** (Z-score threshold = 2.5)
- **Spline interpolation** for missing data (30-frame window ≈ 1 second)
  - This process is **repeated 3 times** to ensure stable tracking
- **Pattern-based filling** for occlusion (uses 5 frames before/after)
  - Can be skipped if occlusion is not expected
  - Uses connection chains to prioritize consistent keypoint visibility:
    - Nose → Right Eye  
    - Nose → Left Eye  
    - Neck → Left Shoulder  
    - Neck → Right Shoulder  
    - Right Shoulder → Right Elbow  
    - Right Elbow → Right Wrist  
    - Left Shoulder → Left Elbow  
    - Left Elbow → Left Wrist  
- **Low-pass filter**: Butterworth (cutoff = 10 Hz, `numtaps = 31`)
- **Hip position estimation** (required for OpenSim model scaling)
- Output: **`trc` file** of filtered 3D keypoints

### 5️⃣ Marker Augmentation
- Run OpenCap's **LSTM-based augmentation** to estimate additional elbow and wrist positions

### 6️⃣ Model Scaling
- Scale the OpenSim model using the **most static segment** of the video

### 7️⃣ Inverse Kinematics (IK)
- Run OpenSim’s **IK tool** using the scaled model and augmented `.trc` file

---

## 🧩 OpenSim Model

This pipeline uses a **modified MASI model** ([MASI link](https://simtk.org/frs/?group_id=982)):

- Two **3-DOF joints at the neck** (improved head/neck motion capture)
- **Lower extremities removed**, since lower-body tracking is not possible in occluded/seated conditions

Model file used: `MASI-upperOnly-markers-slider_noEars_scaled_no_patella.osim`

---

## 🧪 Visualizing Results in OpenSim

After running the full pipeline, the following folders will be created:
```
/Model/
└── MASI-upperOnly-markers-slider_noEars_scaled_no_patella.osim

/Kinematics/
└── keypoints_augmented.mot
```


To visualize:
1. Open OpenSim
2. Load the `.osim` model from the **Model** folder
3. Load the `.mot` motion file from the **Kinematics** folder
4. Use OpenSim’s tools to inspect joint trajectories, segment orientations, or export kinematic data

---

## 🙏 Acknowledgements

FrankenCap is built upon the exceptional [OpenCap-core](https://github.com/stanfordnmbl/opencap-core) project by the Stanford [Neuromuscular Biomechanics Lab](https://nmbl.stanford.edu/).  
We are deeply grateful to the Stanford team for developing and sharing such a robust and extensible tool for markerless motion capture.

---

## 📬 Contact

Have questions or suggestions?  
Feel free to [open an issue](https://github.com/your-org/frankencap/issues) or contact the project maintainers directly.
