# Diff-gaussian-rasterization w.r.t camera pose (4×4 Transformation Matrix)
This is the official implementation of diff-gaussian-rasterization module in <a href="https://github.com/hjr37/CG-SLAM">CG-SLAM</a>.
# Derivation Framework
We have decomposed the entire pose derivation process from <strong>top to bottom</strong>, which is clearly visualized in the following figures. More detailed results will come soon!
## Overview
<p align="center">
  <img src="./assets/derivation1.png" hspace="20"/>
</p>

## Color Branch
<p align="center">
  <img src="./assets/derivation2.png" hspace="20"/>
</p>

## Opacity Branch
<p align="center">
  <img src="./assets/derivation3.png" hspace="20"/>  
</p>

# Diff-gaussian-rasterization-Full
## Running Code


<p style="text-align: justify;">This is the method to invoke our <strong>diff-gaussian-rasterization</strong> library.</p>

```python
render(viewpoint_cam, self.gaussians, self.pipe_hyper, self.background, viewmatrix=w2cT, fov=(self.half_tanfovx, self.half_tanfovy), HW=(self.H, self.W), gt_depth=gt_depth, track_off=True, map_off=False)
```



## output
- The output of `GasussianRasterizer`:
```
    --render: rendered_image,
    --viewspace_points: screenspace_points,
    --visibility_filter: radii > 0,
    --radii: radii,
    --depth: depth, 
    --depth_median: depth_median,
    --opacity_map: opacity_map,
    --depth_var: depth_var,
    --gau_uncertainty: gau_uncertainty,
    --num_related_pixels: gau_related_pixels
```

# Diff-gaussian-rasterization-Light (Ignore )
