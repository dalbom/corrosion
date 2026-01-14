# Post Image Processing Methods (IMGPROC.md)

This document provides detailed explanations of the color/intensity correction methods available for post-processing generated corrosion images.

## Overview

Generated images from the diffusion model often exhibit systematic intensity offsets compared to real images. This is because the model learns a distribution that may not perfectly align with the ground truth distribution. Post-processing correction methods help bridge this gap by transforming the intensity values of generated images to better match real images.

All methods operate on the **RED channel only**, as corrosion intensity is encoded in this channel.

---

## Available Methods

### 1. Histogram Matching

**Concept:** Transforms the histogram of the generated image to match the histogram of the reference (real) images.

**How it works:**
1. Compute the Cumulative Distribution Function (CDF) of both source (generated) and target (real) images
2. For each intensity value in the source, find the corresponding intensity in the target that has the same cumulative probability
3. Build a Look-Up Table (LUT) mapping source intensities to target intensities

**Mathematical formulation:**
```
For each pixel value v in source image:
    new_value = CDF_target^(-1)(CDF_source(v))
```

**Pros:**
- Comprehensive distribution matching
- Works well when distributions have similar shapes but different positions

**Cons:**
- May introduce artifacts if distributions are very different in shape
- Can amplify noise in low-probability regions

**Use case:** General-purpose correction, good baseline method.

---

### 2. Offset Correction

**Concept:** Adds or subtracts a constant value to shift the mean intensity.

**How it works:**
```python
offset = mean(real_pixels) - mean(generated_pixels)
corrected_pixel = generated_pixel + offset
```

**Pros:**
- Simplest method, very fast
- Preserves relative differences between pixels

**Cons:**
- Only corrects the mean, doesn't address distribution shape differences
- May clip values at 0 or 255

**Use case:** When generated images are simply too bright or too dark overall.

---

### 3. Scaling Correction

**Concept:** Multiplies pixel values by a factor to match mean intensity.

**How it works:**
```python
scale = mean(real_pixels) / mean(generated_pixels)
corrected_pixel = generated_pixel * scale
```

**Pros:**
- Preserves relative ratios between pixels
- Simple and fast

**Cons:**
- Only corrects the mean through multiplication
- Can cause significant clipping at 255
- Assumes zero remains zero

**Use case:** When generated images have proportionally different intensities.

---

### 4. Linear Regression

**Concept:** Learns a linear transformation (y = mx + b) that best maps generated values to real values.

**How it works:**
1. Sample corresponding pixels from generated and real images
2. Fit a linear regression: `real = slope * generated + intercept`
3. Apply the transformation to all generated pixels

**Pros:**
- Combines offset and scaling in one transformation
- Statistically optimal for linear relationships

**Cons:**
- Assumes linear relationship between distributions
- May not capture complex nonlinear patterns

**Use case:** When the relationship between generated and real intensities is approximately linear.

---

### 5. Soft Histogram Matching

**Concept:** Blends histogram-matched result with the original image for a softer correction.

**How it works:**
```python
matched = histogram_match(generated, real)
corrected = alpha * matched + (1 - alpha) * generated
```

Where `alpha = 0.5` by default.

**Pros:**
- Preserves some original image characteristics
- Less aggressive than full histogram matching
- Can reduce artifacts from histogram matching

**Cons:**
- May not fully correct distribution differences
- Requires tuning of alpha parameter

**Use case:** When full histogram matching is too aggressive or introduces artifacts.

---

### 6. Nonlinear Curve Fitting (Gamma + Polynomial LUT)

**Concept:** Applies gamma correction followed by a polynomial Look-Up Table.

**How it works:**
1. **Gamma correction:** `corrected = scale * (pixel / 255)^gamma * 255`
   - Parameters (gamma, scale) are fitted to minimize error between generated and real
2. **Polynomial LUT:** Fits a 3rd-degree polynomial to the CDF mapping for additional refinement

**Pros:**
- Can capture nonlinear relationships
- Gamma correction is physically meaningful (models monitor response)

**Cons:**
- More complex, may overfit
- Gamma fitting can be unstable

**Use case:** When the relationship between distributions is clearly nonlinear.

---

### 7. Optimal Transport (Earth Mover's Distance)

**Concept:** Finds the mathematically optimal way to transform one distribution into another by minimizing the "transport cost" (Earth Mover's Distance).

**How it works:**
1. Compute histograms of both distributions as probability masses
2. Define a cost matrix where `cost[i][j]` is the squared distance between intensity i and j
3. Solve the optimal transport problem to find the minimum-cost mapping
4. Build a LUT where each source intensity maps to the weighted average of its target intensities

**Mathematical formulation:**
```
Minimize: Σ_ij T_ij * C_ij
Subject to: Σ_j T_ij = p_source[i]  (row constraint)
            Σ_i T_ij = p_target[j]  (column constraint)
            T_ij >= 0
```

Where T is the transport plan, C is the cost matrix, and p are the probability distributions.

**Pros:**
- Theoretically optimal in terms of Wasserstein distance
- Produces smooth, natural-looking mappings
- More robust to outliers than histogram matching

**Cons:**
- Computationally more expensive (O(n³) for n bins)
- Requires the POT (Python Optimal Transport) library

**Use case:** When you want the mathematically optimal correction, especially for distributions with different shapes.

---

## Comparison of Methods

| Method | Corrects Mean | Corrects Shape | Preserves Structure | Complexity |
|--------|---------------|----------------|---------------------|------------|
| Histogram Matching | ✓ | ✓ | Partially | Low |
| Offset | ✓ | ✗ | ✓ | Lowest |
| Scaling | ✓ | ✗ | ✓ | Lowest |
| Linear Regression | ✓ | Partially | ✓ | Low |
| Soft Histogram | ✓ | Partially | ✓ | Low |
| Nonlinear Fitting | ✓ | ✓ | Partially | Medium |
| Optimal Transport | ✓ | ✓ | ✓ | Higher |

---

## Usage

### Applying Corrections

```bash
python correct_generated_images.py
```

This interactive script will:
1. Display a menu of available methods
2. Load generated images from `generated/` directory
3. Apply the selected correction method
4. Save corrected images to `corrected/` directory
5. Print before/after error statistics

### Visualizing Results

```bash
python create_histogram_figure.py
```

This creates a KDE (Kernel Density Estimate) plot showing:
- **Red (filled):** Real image distribution
- **Blue (dashed):** Original generated distribution
- **Green (solid):** Corrected distribution

---

## Implementation Details

All correction functions are implemented in `correct_generated_images.py`. Key functions:

- `get_histogram_mapping()` - Computes histogram matching LUT
- `get_optimal_transport_lut()` - Computes OT-based LUT
- `fit_gamma_curve()` - Fits gamma correction parameters
- `fit_polynomial_lut()` - Fits polynomial LUT
- `apply_correction_and_save()` - Applies any method and saves result

---

## References

1. **Histogram Matching:** Gonzalez & Woods, "Digital Image Processing"
2. **Optimal Transport:** Peyré & Cuturi, "Computational Optimal Transport" (2019)
3. **POT Library:** https://pythonot.github.io/
