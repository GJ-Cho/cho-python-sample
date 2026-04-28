# Touch Pose Estimation — Calibration Board / ArUco Marker

PyQt5 GUI application that detects a Zivid calibration board or ArUco marker from a ZDF file (or live capture), and computes the object's pose in the robot-base frame using a hand-eye calibration matrix.

## Features

- **Detection source toggle**: Calibration Board (checkerboard) or ArUco Marker
- **Camera configuration toggle**: Hand-to-Eye (Eye-to-Hand) or Hand-in-Eye (Eye-in-Hand)
- **Multiple markers**: when more than one ArUco marker is detected, the one with the smallest ID is used first (0 → 1 → 2 …)
- **9 rotation output formats** with robot-brand descriptions
- **Radian / Degree toggle** for display; terminal always prints both units on separate lines
- **6DoF result panel** with a distinct green-on-dark colour for easy reading
- **3D visualisation** via Open3D (background thread, coordinate frames at detected pose)

## Requirements

- Zivid SDK with Python bindings (`zivid`)
- `zividsamples` helper package (from this repo's `modules/` directory)
- PyQt5, Open3D, NumPy, SciPy

## Usage

```bash
python touch_pose_estimation_cal_board_marker.py
```

### Workflow

1. **Load data** — click *Load ZDF* to open a `.zdf` file, or connect a camera and click *Capture*.
2. **Detection Source** — choose *Calibration Board* or *ArUco Marker*.
   - ArUco: select the dictionary and enter the marker IDs to search for.
3. **Camera Configuration** — choose *Hand-to-Eye* or *Hand-in-Eye*.
4. **Hand-Eye Matrix** — load a YAML file or paste 16 numbers (row-major 4×4).
5. **Robot Capture Pose** *(Hand-in-Eye only)* — load a YAML file or enter the pose manually in any supported format.
6. Click **Detect & Estimate** → results appear in the GUI and are printed to the terminal.
7. Click **Visualize in 3D** to open an Open3D viewer.

## Transformation Formulas

| Mode | Formula |
|------|---------|
| Hand-to-Eye (Eye-to-Hand, camera fixed) | `T_robot = T_hand_eye × T_object_camera` |
| Hand-in-Eye (Eye-in-Hand, camera on robot) | `T_robot = T_capture × T_hand_eye × T_object_camera` |

## Supported Rotation Formats

| Format | Convention | Representative Robots |
|--------|-----------|----------------------|
| Rotation Vector `[rx ry rz]` | Axis-angle magnitude | Universal Robots (PolyScope) |
| Quaternion `[qx qy qz qw]` | Unit quaternion | ABB (RAPID) |
| Euler XYZ extrinsic | Fixed axes X→Y→Z | Fanuc, Motoman/Yaskawa |
| Euler ZYX extrinsic | Fixed axes Z→Y→X | Epson, CRS |
| Euler ZYX intrinsic (A-B-C) | Moving axes Z→Y'→X'' | ABB, KUKA, Nachi |
| Euler ZYZ extrinsic | Fixed axes Z→Y→Z | Denso |
| Euler ZYZ intrinsic | Moving axes Z→Y'→Z'' | Doosan Robotics, Adept, Comau, Kawasaki |
| Euler ZXZ intrinsic | Moving axes Z→X'→Z'' | CATIA, SolidWorks |
| Euler XYZ intrinsic | Moving axes X→Y'→Z'' | Stäubli (VAL3), Mecademic |

> **Convention note** — scipy uses lowercase sequence strings for extrinsic (fixed-axis) rotations and uppercase for intrinsic (moving-axis) rotations. For example, `"zyx"` is extrinsic Z→Y→X and `"ZYX"` is intrinsic Z→Y'→X''.

## Input Formats for Robot Capture Pose

| Selection | Expected input |
|-----------|---------------|
| 4×4 Matrix (File YAML) | YAML file saved by `save_load_matrix` |
| 4×4 Matrix (Manual Text) | 16 space-separated numbers, row-major |
| xyz + &lt;rotation format&gt; | `x y z r1 r2 r3` (mm + rad or deg); quaternion requires 7 values |
