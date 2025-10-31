"""
A small stub that simulates land-cover segmentation using a simple heuristic on RGB values.
Replace this with a real HF U-Net / segmentation model for production.
"""
import numpy as np
import rasterio


def segment_impervious(rgb_array):
"""Return a mask of impervious surfaces (1) vs non-impervious (0).
rgb_array: shape (3, H, W)
"""
r = rgb_array[0].astype(float)
g = rgb_array[1].astype(float)
b = rgb_array[2].astype(float)
# heuristic: high red and low green => impervious (synthetic data)
mask = (r > 120) & (g < 120)
return mask.astype('uint8')


if __name__ == '__main__':
with rasterio.open('data/synthetic_rgb.tif') as src:
arr = src.read() # (3,H,W)
mask = segment_impervious(arr)
with rasterio.open('data/impervious_mask.tif','w',
driver='GTiff',
height=mask.shape[0],
width=mask.shape[1],
count=1,
dtype='uint8',
crs=src.crs,
transform=src.transform) as dst:
dst.write(mask,1)
print('Impervious mask saved to data/impervious_mask.tif')