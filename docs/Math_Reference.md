# NeRF Mathematical Reference

Quick reference for the key mathematical concepts in Neural Radiance Fields and where they're implemented in your code.

## Core Mathematical Concepts

### 1. Ray Parameterization

**Equation**: `r(t) = o + t·d`

Where:
- `o` = ray origin (camera position)
- `d` = ray direction (unit vector)
- `t` = parameter along ray (distance from origin)

**Implementation**: `src/utils/nerf_utils.py` → `get_rays()`

### 2. Volume Rendering Equation

**Continuous form**:
```
C(r) = ∫[t_n to t_f] T(t) · σ(r(t)) · c(r(t), d) dt
```

**Discrete approximation**:
```
C(r) = Σ[i=1 to N] T_i · α_i · c_i
```

Where:
- `C(r)` = final color along ray r
- `T_i = exp(-Σ[j=1 to i-1] σ_j · δ_j)` = transmittance 
- `α_i = 1 - exp(-σ_i · δ_i)` = alpha compositing weight
- `σ_i` = density at sample i
- `c_i` = color at sample i
- `δ_i` = distance between samples

**Implementation**: `src/utils/nerf_utils.py` → `raw2outputs()`

### 3. Positional Encoding

**Equation**:
```
γ(p) = [p, sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), ..., sin(2^(L-1)πp), cos(2^(L-1)πp)]
```

**Purpose**: Map inputs to higher dimensional space to help neural network learn high-frequency functions.

**Implementation**: `src/models/nerf.py` → `positional_encoding()`

### 4. Hierarchical Sampling

**Coarse sampling**: Uniform distribution
```
t_i = t_n + (t_f - t_n) · i/N
```

**Fine sampling**: Importance sampling based on coarse weights
```
PDF(t) ∝ w(t)  where w(t) are weights from coarse network
```

**Implementation**: 
- Coarse: `src/rendering/volume_renderer.py` → `sample_coarse_points()`
- Fine: `src/rendering/volume_renderer.py` → `sample_fine_points()`

## Key Algorithms

### Algorithm 1: Ray Generation
```
Input: Image dimensions (H, W), focal length f, camera pose c2w
Output: Ray origins rays_o, ray directions rays_d

1. Create pixel coordinate grid (i, j)
2. Convert to normalized device coordinates:
   dirs = [(i - W/2)/f, -(j - H/2)/f, -1]
3. Transform to world coordinates:
   rays_d = dirs @ c2w[:3, :3]^T
   rays_o = broadcast(c2w[:3, 3])
```

### Algorithm 2: Volume Rendering
```
Input: Colors c_i, densities σ_i, distances δ_i
Output: Final pixel color C

1. Compute alpha values: α_i = 1 - exp(-σ_i · δ_i)
2. Compute transmittance: T_i = ∏[j=1 to i-1](1 - α_j)
3. Compute weights: w_i = T_i · α_i
4. Integrate: C = Σ[i=1 to N] w_i · c_i
```

### Algorithm 3: Training Step
```
Input: Batch of rays (origins, directions, target colors)
Output: Updated network parameters

1. Sample points along rays
2. Query network for colors and densities
3. Volume render to get predicted colors
4. Compute loss: L = ||C_pred - C_target||²
5. Backpropagate and update weights
```

## Network Architecture Details

### Input Dimensions
- Position encoding: `3 + 3 × 2 × L_pos` (L_pos = 10, so 63 total)
- Direction encoding: `3 + 3 × 2 × L_dir` (L_dir = 4, so 27 total)

### Output Dimensions
- RGB: 3 values (colors)
- Density: 1 value (σ)

### Network Structure
```
Input (63D) → FC(256) → FC(256) → FC(256) → FC(256) → [Skip] → FC(256) → FC(256) → FC(256) → FC(256)
                                                          ↑
                                                    Concatenate input
```

## Loss Functions

### Primary Loss
```
L = ||C_coarse - C_gt||² + ||C_fine - C_gt||²
```

### Metrics
- **MSE**: Mean Squared Error between predicted and ground truth colors
- **PSNR**: Peak Signal-to-Noise Ratio = -10 × log₁₀(MSE)

**Implementation**: `src/utils/nerf_utils.py` → `img2mse()`, `mse2psnr()`

## Coordinate Systems

### Camera Coordinate System
- X: Right
- Y: Up  
- Z: Backward (negative viewing direction)

### World Coordinate System
- Defined by camera poses
- Each camera has transformation matrix c2w (camera-to-world)

### Transformations
```
World point = c2w @ Camera point
Camera ray direction = c2w[:3, :3] @ Camera direction
```

## Implementation Notes

### Memory Optimization
- **Chunk processing**: Process rays in batches to fit in GPU memory
- **Hierarchical sampling**: Focus computation where needed

### Numerical Stability
- **Small epsilon**: Add 1e-10 to prevent log(0) in transmittance
- **Density activation**: Use ReLU on raw density output

### Training Strategies
- **Learning rate decay**: Reduce LR at specific iterations
- **Coarse-to-fine**: Train coarse network first, then fine network
- **Stratified sampling**: Add noise to sample positions during training

## Common Debugging Equations

### Check Ray Generation
```python
# Rays should point away from camera
assert torch.all(rays_d @ camera_forward < 0)  # Assuming forward = [0,0,-1]
```

### Check Volume Rendering
```python
# Weights should sum to approximately 1 (alpha compositing)
weight_sum = torch.sum(weights, dim=-1)
assert torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=0.1)
```

### Check Transmittance
```python
# Transmittance should decrease along ray (monotonically)
T = torch.cumprod(1 - alpha, dim=-1)
assert torch.all(T[..., 1:] <= T[..., :-1] + 1e-6)
```

This mathematical foundation is what makes NeRF work - elegant equations that neural networks can learn to approximate!
