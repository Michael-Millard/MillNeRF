# MillNeRF - Neural Radiance Fields from Scratch

A complete implementation of Neural Radiance Fields (NeRF) built from scratch for educational purposes.

## 🚀 Quick Start

### Installation
```bash
# Clone and setup
git clone <your-repo>
cd MillNeRF

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

### Usage

#### Method 1: Using the main entry point (Recommended)
```bash
# Prepare your data
python src/main.py prepare --images_dir path/to/images --output_dir data

# Train the model
python src/main.py train --config configs/default.yaml

# Render novel views
python src/main.py render --checkpoint build/checkpoints/latest.pth
```

#### Method 2: Using individual scripts
```bash
# Prepare data
python src/data/prepare.py --images_dir path/to/images

# Train model  
python src/training/train.py --config configs/default.yaml

# Render views
python src/rendering/render.py --checkpoint build/checkpoints/latest.pth
```

#### Method 3: After installation with pip
```bash
# If you installed with pip install -e .
millnerf prepare --images_dir path/to/images
millnerf train --config configs/default.yaml
millnerf render --checkpoint build/checkpoints/latest.pth
```

## 📁 Project Structure

```
MillNeRF/
├── src/                    # All source code
│   ├── main.py            # Main entry point
│   ├── models/            # Neural network models
│   ├── data/              # Data loading and preparation
│   ├── rendering/         # Volume rendering and view synthesis
│   ├── training/          # Training loops and utilities
│   └── utils/             # Helper functions
├── tests/                 # Test scripts
├── docs/                  # Documentation
├── configs/               # Configuration files
├── data/                  # Training data (you create this)
├── build/                 # Generated outputs
├── setup.py              # Python package setup
└── requirements.txt       # Dependencies
```

## 🧪 Testing

```bash
# Run tests
python tests/test_nerf.py

# Or test individual components
python -m pytest tests/
```

## 📚 Learning Resources

Check the [`docs/`](docs/) folder for comprehensive learning materials:
- [`docs/NeRF_Tutorial.md`](docs/NeRF_Tutorial.md) - Hands-on tutorial
- [`docs/NeRF_Deep_Dive.md`](docs/NeRF_Deep_Dive.md) - Complete theory
- [`docs/Math_Reference.md`](docs/Math_Reference.md) - Mathematical reference
- [`docs/GETTING_STARTED.md`](docs/GETTING_STARTED.md) - Practical guide

## 🛠️ Development

This project follows standard Python package structure:
- `src/` contains all source code
- `tests/` contains test files  
- `setup.py` defines the package
- Entry points provide clean command-line interface

Perfect for learning, extending, and contributing!