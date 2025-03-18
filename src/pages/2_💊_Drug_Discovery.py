import streamlit as st

st.set_page_config(
    page_title="Drug Discovery",
    page_icon="💊",
)

st.title("Drug Discovery")


with st.sidebar:
    st.header("About")
    st.info("""
    This app allows you to search for proteins using natural language queries,
    visualize their 3D structures, and analyze their properties.
    
    Data sources: UniProt, Pfam
    """)
    
    st.header("Examples")
    st.markdown("""
    - Show proteins related to plastic degradation
    - Search for hydrolase enzymes
    - Find the proteins that cause Alzheimer's 
    """)