# MillNeRF - Getting Started Guide

Congratulations! Your NeRF implementation is now complete and ready to use. Here's everything you need to know to get started.

## 🎯 What You Have

A complete Neural Radiance Fields implementation from scratch including:

- **Core NeRF Model**: Multi-layer perceptrons with positional encoding
- **Hierarchical Sampling**: Coarse and fine networks with importance sampling  
- **Volume Rendering**: Full volumetric rendering pipeline
- **Training Infrastructure**: Complete training loop with validation and checkpointing
- **Data Loading**: Flexible data loading supporting NeRF JSON format
- **Visualization**: Rendering scripts for novel view synthesis

## 📋 Prerequisites

- Python 3.12 (already set up in your virtual environment)
- Around 75 images of your scene from different viewpoints
- CUDA-capable GPU recommended (but CPU works for testing)

## 🚀 Quick Start

### Step 1: Prepare Your Images

1. Place your images in a directory (e.g., `raw_images/`)
2. Images should be taken from various viewpoints around your scene
3. JPG, PNG formats are supported

### Step 2: Prepare Data for Training

```bash
# Resize images and create camera poses
python prepare_data.py --images_dir raw_images/ --output_dir data --resize_factor 4
```

This will:
- Resize your images by factor 4 to save memory
- Generate circular camera poses (dummy poses for now)
- Create a `transforms.json` file
- Place everything in the `data/` directory

### Step 3: Start Training

```bash
# Start training with default configuration
python train.py --config configs/default.yaml
```

Training will:
- Save checkpoints every 50,000 iterations in `build/checkpoints/`
- Log progress every 1,000 iterations
- Render test images every 25,000 iterations in `build/renders/`
- Run for 1,000,000 iterations (you can stop early with Ctrl+C)

### Step 4: Monitor Progress

Check `build/renders/` periodically to see how your NeRF is learning:
- Training typically takes several hours to days depending on scene complexity
- You should see improvements in image quality over time
- PSNR values will be printed during validation

### Step 5: Render Novel Views

```bash
# Render novel viewpoints after training
python render.py --checkpoint build/checkpoints/latest.pth --mode novel
```

This creates a 360-degree video of your scene!

## 📊 Expected Training Times

- **CPU**: Very slow, only for testing (hours for small scenes)
- **GPU (GTX 1060+)**: 2-6 hours for good results  
- **GPU (RTX 3080+)**: 1-3 hours for good results

## 🎛️ Configuration Options

Edit `configs/default.yaml` to customize:

- **Image resolution**: Adjust `data.image_scale` (4 = 1/4 resolution)
- **Training iterations**: Change `training.max_iterations`
- **Learning rate**: Modify `training.learning_rate`
- **Sampling**: Adjust `sampling.num_coarse_samples` and `num_fine_samples`
- **Batch size**: Change `training.batch_size` (reduce if out of memory)

## 🔧 Troubleshooting

### Out of Memory Errors
- Reduce `training.batch_size` in config
- Increase `data.image_scale` (more downsampling)
- Reduce `rendering.chunk_size`

### Poor Results
- You need better camera poses (see Camera Poses section below)
- Try training longer
- Adjust learning rate
- Check that your scene has good coverage from all angles

### Training Too Slow
- Reduce image resolution (`image_scale`)
- Use fewer samples (`num_coarse_samples`, `num_fine_samples`)
- Use a GPU if possible

## 📷 Better Camera Poses (Recommended)

The `prepare_data.py` script generates dummy circular poses, which work for testing but aren't optimal. For best results:

1. **Use COLMAP** (recommended):
   ```bash
   # Install COLMAP and process your images
   colmap feature_extractor --database_path database.db --image_path raw_images/
   colmap exhaustive_matcher --database_path database.db
   colmap mapper --database_path database.db --image_path raw_images/ --output_path sparse/
   ```

2. **Convert COLMAP output** to NeRF format using available conversion scripts

3. **Manual pose estimation** using camera calibration tools

## 📁 Project Structure

```
MillNeRF/
├── data/                     # Your processed data
│   ├── images/              # Training images  
│   └── transforms.json      # Camera poses
├── src/                     # Source code
│   ├── models/             # Neural network models
│   ├── data/               # Data loading
│   ├── rendering/          # Volume rendering
│   ├── training/           # Training loops
│   └── utils/              # Utility functions
├── configs/                # Configuration files
├── build/                  # Generated outputs
│   ├── checkpoints/       # Model checkpoints
│   ├── renders/           # Rendered images
│   └── logs/              # Training logs
├── train.py               # Main training script
├── render.py              # Rendering script
├── prepare_data.py        # Data preparation
└── test_nerf.py          # Tests
```

## 🎯 Next Steps

1. **Collect Your Images**: Take photos of your scene from many angles
2. **Prepare Data**: Run the data preparation script
3. **Start Training**: Begin with the default configuration
4. **Experiment**: Try different settings and scenes
5. **Share Results**: Your first NeRF is ready to amaze people!

## 🔬 Advanced Topics

- **Custom Loss Functions**: Modify the training loop in `src/training/train.py`
- **New Architectures**: Experiment with the model in `src/models/nerf.py`
- **Different Samplers**: Customize sampling in `src/rendering/volume_renderer.py`
- **Multi-GPU Training**: Add DataParallel support
- **Custom Datasets**: Extend the dataset class in `src/data/dataset.py`

## 🐛 Need Help?

If you encounter issues:
1. Run `python test_nerf.py` to verify your installation
2. Check that all dependencies are installed
3. Start with smaller images and fewer iterations for testing
4. Review the console output for error messages

Happy NeRF-ing! 🎉
