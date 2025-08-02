# NeRF Implementation Tutorial: From Theory to Code

This tutorial walks you through the key concepts behind NeRF and shows you exactly where each concept is implemented in your codebase. Perfect for understanding both the "why" and the "how" of Neural Radiance Fields.

## Quick Concept Overview

**What NeRF does**: Takes photos of a scene from different angles → Creates a 3D representation → Generates new photos from any angle

**How it works**: Neural network learns to predict color and opacity at any 3D point → Volume rendering combines these predictions into images

## Core Concepts & Code Locations

### 1. 🎯 Ray Generation: Connecting 2D Pixels to 3D Space

**Concept**: For each pixel in a photo, create a ray that goes from the camera through that pixel into the 3D world.

**Why important**: This is how we connect 2D training images to 3D scene understanding.

**Code**: `src/utils/nerf_utils.py` → `get_rays()` function
```python
# Key insight: Transform pixel coordinates to world space directions
dirs = torch.stack([(i - W * 0.5) / focal,    # X direction
                   -(j - H * 0.5) / focal,     # Y direction  
                   -torch.ones_like(i)], -1)   # Z direction (forward)

# Apply camera rotation to get world space rays
rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
```

**Used in**: `src/data/dataset.py` → `_generate_training_rays()`

---

### 2. 📍 Point Sampling: Where to Look Along Each Ray

**Concept**: Along each ray, sample multiple 3D points where we'll ask the neural network "what's here?"

**Why important**: We need to check many depths to understand what's visible vs. hidden.

**Two strategies**:
- **Coarse sampling**: Evenly spaced points along the ray
- **Fine sampling**: More points where surfaces are likely to be

**Code**: `src/rendering/volume_renderer.py`
- `sample_coarse_points()` - Uniform sampling
- `sample_fine_points()` - Importance sampling based on where coarse network found surfaces

```python
# Coarse: Sample uniformly between near and far planes
z_vals = self.near * (1. - t_vals) + self.far * t_vals

# Fine: Sample more where coarse network found surfaces
# (Uses probability distribution from coarse weights)
```

---

### 3. 🌊 Positional Encoding: Helping Neural Networks See Details

**Concept**: Transform 3D coordinates using sine/cosine functions before feeding to the network.

**Why needed**: Neural networks naturally learn smooth functions. To capture sharp edges and fine textures, we need to "frequency-encode" the inputs.

**Mathematical transformation**:
```
[x, y, z] → [x, y, z, sin(x), cos(x), sin(2x), cos(2x), sin(4x), cos(4x), ...]
```

**Code**: `src/models/nerf.py` → `positional_encoding()` method
```python
for i in range(L):
    freq = 2.0 ** i
    encoded.append(torch.sin(freq * torch.pi * x))
    encoded.append(torch.cos(freq * torch.pi * x))
```

**Used for**: Both 3D positions (10 frequency levels) and viewing directions (4 levels)

---

### 4. 🧠 Neural Network: The Scene Memory

**Concept**: A neural network that memorizes the entire scene by learning to map:
- **Input**: 3D position + viewing direction
- **Output**: Color (R,G,B) + density (how solid/transparent)

**Architecture details**:
- 8 hidden layers with 256 neurons each
- Skip connection at layer 4 (helps training)
- Separate processing for position vs. viewing direction

**Code**: `src/models/nerf.py` → `NeRFModel` class
```python
# Position-dependent features (geometry)
for i, layer in enumerate(self.pos_layers):
    if i in self.skip_connections:
        x = torch.cat([x, input_pts], -1)  # Skip connection
    x = F.relu(layer(x))

# View-dependent color (handles reflections)
color_input = torch.cat([features, dirs_encoded], -1)
```

**Two networks**: Coarse (rough understanding) + Fine (detailed refinement)

---

### 5. 🎨 Volume Rendering: From 3D Points Back to 2D Pixels

**Concept**: Given colors and densities at points along a ray, compute the final pixel color by simulating how light travels through the volume.

**Key insight**: Like looking through colored glass - each point along the ray contributes color, but points behind opaque objects are hidden.

**The math**: 
```
Final_Color = Σ (Transparency_to_point × Opacity_at_point × Color_at_point)
```

**Code**: `src/utils/nerf_utils.py` → `raw2outputs()` function
```python
# Calculate how opaque each sample is
alpha = 1. - torch.exp(-torch.relu(raw[..., 3]) * dists)

# Calculate how much light reaches each point (transmittance)
transmittance = torch.cumprod(1. - alpha + 1e-10, -1)

# Combine everything: final color is weighted sum
weights = alpha * transmittance
rgb_map = torch.sum(weights[..., None] * rgb, -2)
```

**Used in**: `src/rendering/volume_renderer.py` → `render_rays()`

---

### 6. 🎯 Training Loop: Learning from Photos

**Concept**: 
1. Sample random rays from training images
2. Render those rays using current network
3. Compare predicted colors to actual pixel colors
4. Update network to reduce errors

**Code**: `src/training/train.py` → `train_step()` method
```python
# Render rays using current network
outputs = self.renderer.render_rays(self.model, rays_o, rays_d)

# Compare with ground truth
loss_coarse = img2mse(outputs['rgb_coarse'], target_rgb)
loss_fine = img2mse(outputs['rgb_fine'], target_rgb)

# Update network
total_loss.backward()
self.optimizer.step()
```

**Key insight**: We only supervise with 2D images, but the network learns 3D structure!

---

## The Magic: How It All Fits Together

### Training Phase
```
Input Photos → Generate Random Rays → Sample 3D Points → Query NeRF Network 
    ↓
Get Colors & Densities → Volume Render → Compare with True Pixel Colors → Update Network
```

### Novel View Synthesis
```
New Camera Position → Generate All Rays for Image → Sample Points → Query Trained Network
    ↓  
Volume Render → New Photo from Unseen Angle!
```

## Key Files and Their Roles

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/models/nerf.py` | Neural network architecture | `NeRFModel`, `positional_encoding()` |
| `src/rendering/volume_renderer.py` | Volume rendering pipeline | `render_rays()`, `sample_coarse_points()` |
| `src/utils/nerf_utils.py` | Core utilities | `get_rays()`, `raw2outputs()` |
| `src/data/dataset.py` | Data loading and ray generation | `_generate_training_rays()` |
| `src/training/train.py` | Training orchestration | `train_step()`, validation |
| `train.py` | Main training script | Entry point |
| `render.py` | Novel view synthesis | Image and video generation |

## Understanding the Training Process

### What Happens During Training

**Early iterations (0-50k)**:
- Network learns basic scene layout
- Blurry but recognizable shapes
- Coarse geometry understanding

**Mid training (50k-200k)**:
- Details start appearing
- Better color accuracy
- Fine network adds precision

**Late training (200k+)**:
- High-quality textures
- View-dependent effects (reflections)
- Photorealistic results

### Monitoring Your Training

**Check these files**:
- `build/renders/test_render_*.png` - Visual progress
- Console output - Loss values and PSNR
- `build/checkpoints/` - Saved model weights

**Good signs**:
- Steadily decreasing loss
- Increasing PSNR (>20 is good, >30 is excellent)
- Sharper details in test renders

## Experimentation Ideas

Now that you understand the code, try modifying:

1. **Network architecture** (`src/models/nerf.py`):
   - Change hidden layer sizes
   - Modify positional encoding levels
   - Add/remove skip connections

2. **Sampling strategy** (`src/rendering/volume_renderer.py`):
   - Adjust number of coarse/fine samples
   - Experiment with sampling distributions

3. **Training parameters** (`configs/default.yaml`):
   - Learning rates
   - Batch sizes
   - Loss weights

4. **Data processing** (`src/data/dataset.py`):
   - Image preprocessing
   - Ray sampling strategies

## Common Issues and Solutions

**Blurry results**: 
- Increase positional encoding levels
- Train longer
- Check camera poses accuracy

**Out of memory**:
- Reduce batch size in config
- Decrease chunk size for rendering
- Use smaller images

**Slow convergence**:
- Adjust learning rate
- Check data quality
- Verify camera poses

## Next Steps

1. **Run your first NeRF**: Follow the getting started guide
2. **Experiment**: Try different scenes and settings
3. **Extend**: Add features like dynamic scenes or faster training
4. **Share**: Show off your photorealistic novel views!

The beauty of NeRF is its simplicity - just a neural network learning to see in 3D. But as you can see from the code, there's elegant engineering behind that simplicity!
