import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="Smart Supplier Advisor", layout="wide")

st.title("🚀 Smart Supplier Advisor")
st.write("Find the best supplier for your product using AI and your custom criteria!")

# ----- Supplier Data -----
data = [
    {"name":"Supplier A","product":"Widget","price":100,"reliability":8,"delivery_time":5,"location":"France","sustainability":7},
    {"name":"Supplier B","product":"Gadget","price":120,"reliability":9,"delivery_time":6,"location":"Germany","sustainability":6},
    {"name":"Supplier C","product":"Widget","price":90,"reliability":7,"delivery_time":4,"location":"Italy","sustainability":8},
    {"name":"Supplier D","product":"Gadget","price":110,"reliability":6,"delivery_time":7,"location":"Spain","sustainability":9},
    {"name":"Supplier E","product":"Widget","price":95,"reliability":9,"delivery_time":5,"location":"France","sustainability":8},
]
suppliers = pd.DataFrame(data)

# ----- Load model -----
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
model = load_model()

# ----- User input -----
product_input = st.text_input("Product to source:", "")
preferred_location = st.selectbox("Preferred location:", ["Any"] + suppliers['location'].unique().tolist())

st.write("Adjust importance of each criterion (0 = low, 10 = high):")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    weight_price = st.slider("Price", 0, 10, 5)
with col2:
    weight_reliability = st.slider("Reliability", 0, 10, 5)
with col3:
    weight_delivery = st.slider("Delivery time", 0, 10, 5)
with col4:
    weight_location = st.slider("Location", 0, 10, 5)
with col5:
    weight_sustainability = st.slider("Sustainability", 0, 10, 5)

if st.button("Find best suppliers"):
    if product_input.strip() == "":
        st.warning("Please enter a product name.")
    else:
        # Semantic similarity
        product_emb = model.encode(product_input, convert_to_tensor=True)
        supplier_embs = model.encode(suppliers['product'].tolist(), convert_to_tensor=True)
        similarity_scores = util.cos_sim(product_emb, supplier_embs)[0].cpu().numpy()
        suppliers['similarity'] = similarity_scores

        # Weighted score
        suppliers['score'] = (
            weight_price * (1 / suppliers['price']) +
            weight_reliability * suppliers['reliability'] +
            weight_delivery * (1 / suppliers['delivery_time']) +
            weight_location * suppliers['location'].apply(lambda x: 1 if preferred_location=="Any" or x==preferred_location else 0) +
            weight_sustainability * suppliers['sustainability'] +
            weight_price * suppliers['similarity']
        )

        # Show top suppliers
        top_suppliers = suppliers.sort_values(by='score', ascending=False).head(5)
        st.subheader("🏆 Top Suppliers")
        st.dataframe(top_suppliers[['name','product','price','reliability','delivery_time','location','sustainability','score']])
