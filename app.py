"""
Crypto Metrics Dashboard - Main Entry
"""
import streamlit as st

from src.config import APP_TITLE, APP_ICON, APP_LAYOUT, WELCOME_TITLE, WELCOME_TEXT, SIDEBAR_INFO

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout=APP_LAYOUT,
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f'<h1 class="main-header">{WELCOME_TITLE}</h1>', unsafe_allow_html=True)

st.markdown(WELCOME_TEXT)

st.sidebar.info(SIDEBAR_INFO)
