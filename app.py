import streamlit as st
import openai

st.set_page_config(page_title="Smart Supplier Advisor", layout="wide")
st.title("🌐 Smart Supplier Advisor (Global Suppliers)")

st.write("Find real suppliers worldwide using AI! Enter your product and preferences.")

# ----- User inputs -----
product_input = st.text_input("Product to source:", "")
preferred_region = st.selectbox("Preferred region:", ["Any", "Europe", "Asia", "North America", "South America", "Africa"])
weight_price = st.slider("Importance of Price", 0, 10, 5)
weight_reliability = st.slider("Importance of Reliability", 0, 10, 5)
weight_delivery = st.slider("Importance of Delivery Time", 0, 10, 5)
weight_sustainability = st.slider("Importance of Sustainability", 0, 10, 5)
weight_certification = st.slider("Importance of Certification / Quality", 0, 10, 5)

openai_api_key = st.text_input("OpenAI API Key:", type="password")

if st.button("Find suppliers"):
    if not product_input or not openai_api_key:
        st.warning("Please enter product and OpenAI API Key.")
    else:
        openai.api_key = openai_api_key

        prompt = f"""
        You are an expert sourcing assistant. Provide a list of 5 real suppliers worldwide for the product "{product_input}".
        Include the following columns in JSON format: 
        name, country, estimated_price_usd, reliability_score_1_to_10, delivery_time_days, sustainability_score_1_to_10, certification_quality_score_1_to_10.

        Consider these importance weights for ranking: 
        Price={weight_price}, Reliability={weight_reliability}, Delivery={weight_delivery}, Sustainability={weight_sustainability}, Certification={weight_certification}.
        Prefer suppliers in the region: {preferred_region}.
        Only respond with valid JSON array.
        """

        with st.spinner("Querying suppliers..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"user","content":prompt}],
                temperature=0.3,
                max_tokens=600
            )

        try:
            import json
            suppliers = json.loads(response.choices[0].message.content)
            st.subheader("🏆 Top Suppliers")
            st.dataframe(suppliers)
        except Exception as e:
            st.error("Failed to parse response from OpenAI. Here's raw output:")
            st.code(response.choices[0].message.content)
