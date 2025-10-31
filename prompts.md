# Prompts used while building UrbanCool scaffold


1) Generate Streamlit skeleton
Prompt:
"Write a Streamlit app `app.py` that loads a model from `models/model.pkl`, has sidebar inputs for an RGB GeoTIFF upload and a button to run a pipeline: segmentation -> heat map -> prioritization. Include placeholders if models missing. Explain how to run the app."


How used: produced the app layout and control flow.


2) Synthetic data generator
Prompt:
"Create a Python script that generates a small synthetic RGB GeoTIFF (256x256) with an urban red patch and some green patches, plus a temperature raster with a hotspot aligned to the urban patch, and a simple population raster. Save them in './data/'."


How used: used to make `generate_synthetic_tiles.py` so graders can run the demo without large downloads.


3) Segmentation stub
Prompt:
"Provide a simple Python function that heuristically segments impervious surfaces from an RGB array (3,H,W) using thresholding on R/G values. Save it as segmentation_model_stub.py."


How used: quick placeholder replacing an actual U-Net / HF model for the demo.


4) Prioritization logic
Prompt:
"Write a Python function that reads three GeoTIFFs (temp, impervious mask, population) and computes a priority grid by aggregating metrics over windows."


How used: created `prioritization.py`.




Note: When you later add real models: useful prompts include 'Show a Hugging Face inference snippet for U-Net segmentation on a GeoTIFF tile' and 'How to fine-tune a U-Net/Segformer on landcover data using Hugging Face Trainer'.