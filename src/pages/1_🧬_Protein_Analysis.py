import os
import streamlit as st
from rdkit import Chem

from logic.protein_analysis_logic import analyze_protein_with_gemini, display_protein_structure, generate_protein_structure, generate_visual_graphein, query_proteins, validate_pdb

GRAPH_DIR = "protein_graphs"
PDB_FILE = os.path.join(GRAPH_DIR, "protein.pdb")

os.makedirs(GRAPH_DIR, exist_ok=True)

st.set_page_config(
    page_title="Protein Analysis",
    page_icon="🧬",
)

if 'results' not in st.session_state:
    st.session_state['results'] = None
if 'selected_protein' not in st.session_state:
    st.session_state['selected_protein'] = None
if 'protein_analysis' not in st.session_state:
    st.session_state['protein_analysis'] = None
if 'pdb_data' not in st.session_state:
    st.session_state['pdb_data'] = None

st.title("🧬 Protein Analysis App")
st.subheader("Search, Visualize, and Analyze Proteins")

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

query_input = st.text_area("Enter your protein query:", 
                           "Find the proteins that have the Pfam domain PF07224, which is related to plastic degradation.",
                           height=100)

col1, col2 = st.columns([1, 1])
with col1:
    search_button = st.button("🔍 Search", use_container_width=True)
with col2:
    clear_button = st.button("🔄 Clear Results", use_container_width=True)

if clear_button:
    st.session_state['results'] = None
    st.session_state['selected_protein'] = None
    st.session_state['protein_analysis'] = None
    st.session_state['pdb_data'] = None
    st.rerun()

if search_button:
    st.session_state['protein_analysis'] = None
    with st.spinner("🔍 Searching for proteins..."):
        results = query_proteins(query_input)
        st.session_state['results'] = results

if st.session_state['results'] is not None:
    results = st.session_state['results']
    if not results.empty:
        st.success(f"Found {len(results)} proteins")
        
        st.subheader("Results")
        
        display_df = results[['protein_id', 'protein_name', 'organism']].copy()
        display_df.columns = ['ID', 'Protein Name', 'Organism']
        
        st.dataframe(display_df, use_container_width=True)
        
        protein_options = results['protein_id'].tolist()
        selected_protein_id = st.selectbox(
            "Select a protein to analyze:",
            options=protein_options,
            index=0 if protein_options else None
        )
        
        if selected_protein_id:
            st.session_state['selected_protein'] = selected_protein_id
            
            protein_data = results[results['protein_id'] == selected_protein_id].iloc[0].to_dict()
            
            tab1, tab2, tab3 = st.tabs(["📊 Info", "🧬 Sequence", "🔬 3D Structure"])
            
            with tab1:
                st.subheader(f"{protein_data['protein_name']}")
                st.markdown(f"**Organism:** {protein_data['organism']}")
                
                analyze_button = st.button("🔍 Analyze Protein", use_container_width=True)
                
                if analyze_button or st.session_state.get('protein_analysis') is not None:
                    if analyze_button or (st.session_state['selected_protein'] == selected_protein_id and 
                                         st.session_state['protein_analysis'] is None):
                        with st.spinner("Generating analysis..."):
                            analysis = analyze_protein_with_gemini(protein_data)
                            st.session_state['protein_analysis'] = analysis
                    
                    if st.session_state['protein_analysis']:
                        st.markdown(st.session_state['protein_analysis'])
            
            with tab2:
                sequence = protein_data['sequence']
                st.subheader("Amino Acid Sequence")
                
                formatted_sequence = "\n".join([sequence[i:i+80] for i in range(0, len(sequence), 80)])
                st.text(formatted_sequence)
                
                sequence_file = f"{selected_protein_id}.fasta"
                fasta_content = f">{selected_protein_id}|{protein_data['protein_name']}|{protein_data['organism']}\n{sequence}"
                
                st.download_button(
                    label="⬇️ Download FASTA",
                    data=fasta_content,
                    file_name=sequence_file,
                    mime="text/plain"
                )
            
            with tab3:
                if "last_selected_protein" not in st.session_state or st.session_state["last_selected_protein"] != selected_protein_id:
                    st.session_state["pdb_data"] = None
                    st.session_state["last_selected_protein"] = selected_protein_id

                st.subheader("3D Protein Structure")
                st.text(st.session_state['selected_protein'])
                
                structure_button = st.button("🔄 Generate 3D Structure", use_container_width=True)
                
                if structure_button or st.session_state.get('pdb_data') is not None:
                    if structure_button or (st.session_state['selected_protein'] == selected_protein_id and 
                                           st.session_state['pdb_data'] is None):
                        with st.spinner("Predicting protein structure..."):
                            pdb_data = generate_protein_structure(protein_data['sequence'], selected_protein_id)
                            if pdb_data:
                                st.session_state['pdb_data'] = pdb_data
                            else:
                                st.error("⚠️ No PDB structure found for this protein.")
                    
                    if st.session_state.get('pdb_data'):
                        prot_data = st.session_state['pdb_data']
                       
                        # with open(PDB_FILE, 'w') as temp_file:
                        #     temp_file.write(prot_data)
                        # validate_pdb()
                        # g = generate_visual_graphein(PDB_FILE)
                        # fig = plotly_protein_structure_graph(g, node_size_multiplier=1)
                        # st.plotly_chart(fig)
                        structure_html = display_protein_structure(st.session_state['pdb_data'])
                        st.components.v1.html(structure_html._make_html(), height=500)
                        
                        st.download_button(
                            label="⬇️ Download PDB File",
                            data=st.session_state['pdb_data'],
                            file_name=f"{selected_protein_id}.pdb",
                            mime="text/plain"
                        )
                
                st.info("Note: This is a predicted structure and may not represent the actual protein conformation.")
    else:
        st.warning("No proteins found matching your query. Try a different search term.")