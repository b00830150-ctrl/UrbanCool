"""


os.makedirs('data', exist_ok=True)


# create synthetic RGB tile (3 bands) 256x256
w = h = 256
transform = from_origin(2.0, 49.0, 0.0001, 0.0001) # arbitrary affine
rgb = np.zeros((3, h, w), dtype=np.uint8)
# urban center: bright red-ish patch
x0,y0 = 100, 80
rgb[0, y0-40:y0+40, x0-40:x0+40] = 200 # R
rgb[1, y0-40:y0+40, x0-40:x0+40] = 80 # G
rgb[2, y0-40:y0+40, x0-40:x0+40] = 60 # B
# vegetated areas: green patches
for cx,cy in [(40,40),(200,180),(60,200)]:
rgb[1, cy-30:cy+30, cx-30:cx+30] = 180
rgb[0, cy-30:cy+30, cx-30:cx+30] = 40


with rasterio.open('data/synthetic_rgb.tif','w',
driver='GTiff',
height=h,
width=w,
count=3,
dtype=rgb.dtype,
crs='EPSG:4326',
transform=transform) as dst:
dst.write(rgb)


# create synthetic temperature raster (float32) - higher over urban patch
temp = np.full((h,w), 28.0, dtype=np.float32)
for yy in range(h):
for xx in range(w):
# gaussian hotspot
dx = xx - x0
dy = yy - y0
temp[yy,xx] += 10.0 * np.exp(-((dx*dx + dy*dy)/(2*30*30)))


with rasterio.open('data/synthetic_temp.tif','w',
driver='GTiff',
height=h,
width=w,
count=1,
dtype=temp.dtype,
crs='EPSG:4326',
transform=transform) as dst:
dst.write(temp, 1)


# create synthetic population raster (int)
pop = np.zeros((h,w), dtype=np.uint16)
pop[y0-30:y0+30, x0-30:x0+30] = 200 # urban core
pop[y0-80:y0-60, x0+60:x0+80] = 50


with rasterio.open('data/synthetic_pop.tif','w',
driver='GTiff',
height=h,
width=w,
count=1,
dtype=pop.dtype,
crs='EPSG:4326',
transform=transform) as dst:
dst.write(pop, 1)


print('Synthetic tiles saved to ./data/')