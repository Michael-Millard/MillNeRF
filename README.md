# MillNeRF - My Neural Radiance Fields (NeRF) Implementation

This project is entirely based on the original NeRF paper: [NeRF: Neural Radiance Fields for View Synthesis](https://arxiv.org/abs/2003.08934) (Mildenhall et al., 2020).
Their source code can be found [here](https://github.com/bmild/nerf).

<div align="center">
  <img src="media/nerf_training_progress.gif" alt="NeRF Training Progress (200k iterations)" width="100%">
  <p><em>NeRF model training progress over 200k iterations</em></p>
</div>

A complete implementation of Neural Radiance Fields (NeRF) built from scratch for learning purposes. Claude did most of the work. I added a couple things here and there. NeRFs are very computationally expensive. I ran this code locally on my laptop with an RTX 4060 (8GB VRAM). The results are poor. Input images had to be < 100px in each dimension to allow higher hyperparameter values (network depth, batch sizes, chunk sizes, etc.). It works alright. Definitely could do with more hyperparameter fine-tuning and testing. I'll add some more screenshots and GIFs later. This code has only been tested in Ubuntu 24.04 (Linux).

This project requires the use of COLMAP for camera pose extraction. Claude was initially using synthetically generated extrinsics prior to COLMAP integration which was camera calibration heresy. I've added COLMAP code (it's in main with its own sub-parser). It does SfM and extracts the camera extrinsics and stores them automatically in the desired format. All COLMAP -> NeRF conversions are done internally. COLMAP uses OpenCV's coordinate frame: +X right, +Y down, +Z forward. NeRF uses OpenGL's: +X right, +Y up, +Z backwards. Coordinate conversions are annoying. 

The interactive viewer (demos/corrected_viewer.py) almost works but needs some love. Coordinate frame conversions for pose extraction is the problem. I'll get to that soon.

## Features

- **COLMAP Integration**: Full Structure-from-Motion pipeline using COLMAP for camera pose estimation
- **Coordinate System Handling**: Automatic conversion between COLMAP (OpenCV) and NeRF (OpenGL) coordinate systems
- **Interactive Viewer**: Real-time novel view synthesis with [`demos/corrected_viewer.py`](demos/corrected_viewer.py)
- **Animated GIF Generation**: Create training progress animations and novel view sequences
- **Multiple Data Formats**: Support for HEIC conversion and various image formats
- **Comprehensive Documentation**: Detailed learning materials and mathematical references 

## Quick Start

### Installation (Linux)
```bash
# Clone and setup
git clone <your-repo>
cd MillNeRF

# Install COLMAP (required for camera pose estimation)
sudo apt update
sudo apt install colmap

# Create virtual environment
python3 -m venv .millnerf-venv
source .millnerf-venv/bin/activate

# Install dependencies
pip install -e .
```

### Requirements
- **Python 3.8+**: I used 3.12
- **COLMAP**: For Structure-from-Motion camera pose estimation
- **CUDA-capable GPU**: Recommended for reasonable training times
- **8GB+ VRAM**: For full-resolution training (can work with less using smaller images)

### Usage

#### Method 1: Complete Pipeline with COLMAP (Recommended)
```bash
# Step 1: Convert HEIC images to JPEG (if needed -> iPhone problem)
python main.py convert --input_dir path/to/heic/images --output_dir path/to/jpeg/images

# Step 2: Run COLMAP Structure-from-Motion to estimate camera poses
python main.py colmap --images_dir path/to/images --output_dir data/colmap

# Step 3: Train the NeRF model
python main.py train --config configs/default.yaml

# Step 4: Render novel views
python main.py render --checkpoint build/checkpoints/latest.pth

# Step 5: Create animated GIF from training progress (optional)
python main.py gif build/renders --output assets/training_progress.gif --fps 5 --max-size 400

# Optional: Debug COLMAP issues if SfM fails
python main.py debug-colmap --database_path data/colmap/database.db --images_dir path/to/images
```

#### Method 2: Using the main entry point (Legacy)
```bash
# Prepare your data (uses synthetic poses - NOT recommended)
python main.py prepare --images_dir path/to/images --output_dir data

# Train the model
python main.py train --config configs/default.yaml

# Render novel views
python main.py render --checkpoint build/checkpoints/latest.pth
```

#### Method 3: Using individual scripts
```bash
# Prepare data
python src/data/prepare.py --images_dir path/to/images

# Train model  
python src/training/train.py --config configs/default.yaml

# Render views
python src/rendering/render.py --checkpoint build/checkpoints/latest.pth
```

#### Method 4: After installation with pip
```bash
# If you installed with pip install -e .
millnerf colmap --images_dir path/to/images --output_dir data/colmap
millnerf train --config configs/default.yaml
millnerf render --checkpoint build/checkpoints/latest.pth
```

## Interactive Viewer

Launch the interactive NeRF viewer for real-time novel view synthesis:

```bash
python demos/corrected_viewer.py --checkpoint build/checkpoints/latest.pth --config configs/default.yaml
```

Features:
- **Training Mode**: Browse through original training poses
- **Novel Mode**: Generate new viewpoints using spherical coordinates
- **Coordinate System**: Proper handling of COLMAP ↔ NeRF coordinate conversions
- **GPU Memory Management**: Automatic fallback for CUDA out-of-memory situations

## GIF Generation

Create animated GIFs from your NeRF renders:

```bash
# Create GIF from training progress renders
python main.py gif build/renders --output assets/training_progress.gif --fps 5

# Create GIF from novel view sequence
python main.py gif path/to/novel/views --fps 15 --max-size 600

# Batch process multiple render directories
python main.py gif experiments/ --batch --fps 10
```

Options:
- `--fps N`: Set frame rate (default: 10 fps)
- `--max-size N`: Resize large images to N pixels max dimension
- `--loop N`: Set loop count (0 = infinite)
- `--quality N`: Optimization quality 1-100
- `--batch`: Process all subdirectories

## Project Structure

```
MillNeRF/
├── src/                   # All source code
│   ├── main.py            # Main entry point
│   ├── models/            # Neural network models
│   ├── data/              # Data loading and preparation
│   ├── rendering/         # Volume rendering and view synthesis
│   ├── training/          # Training loops and utilities
│   └── utils/             # Helper functions
│       ├── colmap_utils.py    # COLMAP integration utilities
│       ├── colmap_debug.py    # COLMAP diagnostics and fixes
│       └── gif_generator.py   # GIF creation utilities
├── demos/                 # Demo scripts and interactive viewer
│   ├── corrected_viewer.py    # Interactive NeRF viewer
│   └── test_nerf.py          # Testing script
├── docs/                  # Documentation
├── configs/               # Configuration files
├── media/                 # Media files for README (GIFs, images)
├── data/                  # Training data (you create this)
│   ├── images/            # Input images
│   ├── colmap/            # COLMAP output (poses, sparse reconstruction)
│   ├── colmap_relaxed/    # Relaxed COLMAP settings (fallback)
│   └── poses/             # Camera poses in NeRF format
├── build/                 # Generated outputs (not committed)
│   ├── checkpoints/       # Model checkpoints
│   ├── logs/              # Training logs
│   └── renders/           # Rendered images
├── setup.py               # Python package setup
├── main.py                # Main entry point
└── requirements.txt       # Dependencies
```

## COLMAP Integration

This implementation uses COLMAP for robust camera pose estimation from your images:

### Coordinate System Conversion
- **COLMAP Convention**: OpenCV coordinate system (+X right, +Y down, +Z forward)
- **NeRF Convention**: OpenGL coordinate system (+X right, +Y up, +Z backward)
- **Automatic Conversion**: All coordinate transformations are handled internally

### Camera Models Supported
- `PINHOLE`: Basic pinhole camera model
- `SIMPLE_PINHOLE`: Simplified pinhole with single focal length
- `OPENCV`: OpenCV distortion model (default)
- `RADIAL`: Radial distortion model

### Troubleshooting COLMAP
If COLMAP fails to reconstruct your scene:

```bash
# Run diagnostics
python main.py debug-colmap --database_path data/colmap/database.db --images_dir path/to/images

# Try with relaxed settings
python main.py colmap --images_dir path/to/images --output_dir data/colmap_relaxed --camera_model SIMPLE_PINHOLE
```

Common issues and solutions:
- **Too few features**: Ensure images have sufficient texture and overlap
- **Poor lighting**: Images should be well-lit with consistent exposure
- **Motion blur**: Use sharp, well-focused images
- **Insufficient overlap**: Ensure 60-80% overlap between adjacent images

## Testing

```bash
# Run tests
python demos/test_nerf.py

# Or test individual components
python -m pytest demos/
```

## Learning Resources

Check the [`docs/`](docs/) folder for comprehensive learning materials:
- [`docs/NeRF_Tutorial.md`](docs/NeRF_Tutorial.md) - Hands-on tutorial
- [`docs/NeRF_Deep_Dive.md`](docs/NeRF_Deep_Dive.md) - Complete theory
- [`docs/Math_Reference.md`](docs/Math_Reference.md) - Mathematical reference
- [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) - Practical guide
All of these were made by Claude. Pretty neat.

## Development

This project follows standard Python package structure:
- `src/` contains all source code
- `tests/` contains test files  
- `setup.py` defines the package
- `main.py` is the main entry point for the project
- Entry points provide clean command-line interface

## References

**NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis**  
Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng  
*ECCV 2020*  
[[Paper]](https://arxiv.org/abs/2003.08934) [[Project Page]](https://www.matthewtancik.com/nerf) [[Code]](https://github.com/bmild/nerf)
