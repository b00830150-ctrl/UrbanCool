"""
Compute prioritization scores for small tiles combining heat intensity, impervious fraction and population.
"""
import rasterio
import numpy as np


def read_raster(path):
with rasterio.open(path) as src:
arr = src.read(1)
profile = src.profile
return arr, profile




def compute_scores(temp_path, imperv_path, pop_path, window=16):
temp, _ = read_raster(temp_path)
imperv, _ = read_raster(imperv_path)
pop, _ = read_raster(pop_path)


H,W = temp.shape
scores = np.zeros((H//window, W//window))
for i in range(0, H, window):
for j in range(0, W, window):
ti = temp[i:i+window, j:j+window]
im = imperv[i:i+window, j:j+window]
po = pop[i:i+window, j:j+window]
# features
mean_temp = np.nanmean(ti)
imperv_frac = np.nanmean(im)
pop_sum = np.nansum(po)
# score: heat intensity * pop * impervious factor
score = (mean_temp) * (1 + imperv_frac) * (1 + (pop_sum/100.0))
scores[i//window, j//window] = score
return scores


if __name__ == '__main__':
s = compute_scores('data/synthetic_temp.tif','data/impervious_mask.tif','data/synthetic_pop.tif')
print('Scores shape', s.shape)
print(s)