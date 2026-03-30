import streamlit as st

st.set_page_config(page_title="AlphaRisk", page_icon="🛡️")
st.title("AlphaRisk")

tab1, tab2 = st.tabs(["Market Data", "Gamble Simulator"])

with tab1:
    st.write("This is tab 1")

with tab2:
    st.write("This is tab 2")
