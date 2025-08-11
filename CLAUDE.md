# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

Install dependencies:
```bash
pip install -r requirements.txt
```

Install the zivid samples module in development mode:
```bash
cd modules
pip install -e .
```

Run tests:
```bash
python -m pytest tests/
```

Run a specific sample:
```bash
python src/zivid/convert_zdf/convert_zdf_file_dir.py
python src/zivid/stitching/stitch_continuously_capture_visulaize.py --settings-path path/to/settings.yml
```

## Code Architecture

This is a collection of Python samples for 3D vision using Zivid cameras, robotics integration, and 2D/3D computer vision tasks.

### Main Structure
- `src/zivid/` - Core Zivid camera samples organized by functionality:
  - `convert_zdf/` - ZDF file format conversion utilities (to PLY, PNG, depth maps, normal maps)
  - `stitching/` - Point cloud stitching for rotating objects using local registration
  - `stitching_multi_camera/` - Multi-camera calibration and stitching
  - `camera/` - Basic camera capture and live streaming
  - `get_camera_intrinsic/` - Camera calibration utilities
  - `4x4_matrix/` - Transformation matrix utilities for pose conversion

- `src/project/` - Application-specific projects:
  - `UR_communication_test/` - Universal Robots communication via RTDE
  - `pose_estimation/` - Computer vision pose estimation

- `modules/zividsamples/` - Reusable utility modules:
  - `gui/` - PyQt5-based GUI components for calibration and visualization
  - Core utilities for calibration, display, transformation matrices, and robot integration

### Key Dependencies
- **Zivid SDK**: Primary 3D camera interface
- **Open3D**: Point cloud processing and visualization
- **OpenCV**: 2D image processing and computer vision
- **PyQt5**: GUI applications for calibration workflows
- **RoboDK**: Robot simulation and control
- **NumPy/SciPy**: Numerical computing

### Common Patterns
- ZDF files are Zivid's native 3D data format containing point clouds and 2D images
- Point cloud stitching uses local registration without pre-alignment
- GUI applications follow PyQt5 patterns with separate widget classes
- Robot integration supports pose transformations and communication protocols
- Utility functions are centralized in the `modules/zividsamples/` package

### Sample Data
The `sample/` directory contains example ZDF files, PLY point clouds, and configuration files for testing.