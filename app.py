"""
col1, col2 = st.columns([1,2])


with col1:
st.header('Inputs')
if st.button('Generate synthetic tiles'):
import subprocess
subprocess.run(['python','generate_synthetic_tiles.py'])
st.success('Synthetic tiles generated in ./data/')


uploaded = st.file_uploader('Or upload a GeoTIFF RGB tile (optional)', type=['tif','tiff'])
st.markdown('Use the default synthetic data for the demo if you do not upload files.')


with col2:
st.header('Map / Visuals')
# load data
try:
rgb_path = 'data/synthetic_rgb.tif'
temp_path = 'data/synthetic_temp.tif'
pop_path = 'data/synthetic_pop.tif'
with rasterio.open(rgb_path) as src:
rgb = src.read()
with rasterio.open(temp_path) as src:
temp = src.read(1)
except Exception as e:
st.error('Run the data generator first: python generate_synthetic_tiles.py')
st.stop()


# segmentation
mask = segment_impervious(rgb)


fig, axs = plt.subplots(1,3, figsize=(12,4))
im0 = axs[0].imshow(np.transpose(rgb, (1,2,0))/255.0)
axs[0].set_title('RGB')
axs[1].imshow(temp, cmap='hot')
axs[1].set_title('Surface temp (°C)')
axs[2].imshow(mask, cmap='gray')
axs[2].set_title('Impervious mask (stub)')
for a in axs: a.axis('off')
st.pyplot(fig)


# compute prioritization
from prioritization import compute_scores
scores = compute_scores(temp_path, 'data/impervious_mask.tif', pop_path, window=16)


st.subheader('Priority scores (grid)')
fig2, ax2 = plt.subplots(figsize=(6,4))
c = ax2.imshow(scores, cmap='inferno')
ax2.set_title('Priority grid (higher = higher priority)')
fig2.colorbar(c, ax=ax2)
st.pyplot(fig2)


st.markdown('### Suggested mitigation actions (generated heuristically)')
st.write('- Plant trees in highest-score grid cells — estimated local cooling ~0.5–2°C per heavily vegetated cell.')
st.write('- Convert high-impervious rooftops to cool roofs or green roofs.')
st.write('- Prioritize interventions where score × population is highest.')


st.markdown('---')
st.markdown('**Reproducibility**: repo contains `generate_synthetic_tiles.py`, `segmentation_model_stub.py` (replace with HF model), and `prioritization.py`. See `prompts.md` for prompts used to generate this scaffold.')