import streamlit as st
from logic.anomaly_detection import *
from models.AnomalyDetection.model import ProteinAnomalyDetector

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🧫",
)

st.title("Anomaly Detection")

if 'sequence' not in st.session_state:
    st.session_state['sequence'] = None
if 'protein_id' not in st.session_state:
    st.session_state['protein_id'] = None

# Create tabs
tab1, tab2, tab3 = st.tabs(["Sequence Analysis", "Structure Analysis", "ML Anomaly Detection"])

with tab1:
    st.header("Sequence-based Anomaly Detection")
    
    input_method = st.radio("Select input method:", ["UniProt ID", "Paste sequence"])
    
    sequence = None
    if input_method == "UniProt ID":
        uniprot_id = st.text_input("Enter UniProt ID:", "P01308")
        if st.button("Fetch Sequence"):
            with st.spinner("Fetching sequence..."):
                sequence = fetch_uniprot_data(uniprot_id)
                st.session_state['protein_id'] = uniprot_id
                if sequence:
                    st.success(f"Retrieved sequence of length {len(sequence)}")
                    st.session_state['sequence'] = sequence
                    st.code(sequence)
                else:
                    st.error("Failed to retrieve sequence")
    else:
        sequence = st.text_area("Enter protein sequence:", "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN")
        st.session_state['sequence'] = sequence
    
    if st.session_state["sequence"]:
        if st.button("Detect Anomalies"):
            with st.spinner("Analyzing sequence..."):
                results = detect_anomaly_AE_model(
                    st.session_state['sequence'],
                    st.session_state['protein_id']
                )

            if results["is_anomalous"].loc[results.index[0]]:
                st.subheader("Anomaly Detected!")
                target = None
                if st.session_state['protein_id']:
                    target = st.session_state['protein_id']
                else:
                    target = st.session_state['sequence']
                st.text(f"Target Protein : {target}")
                st.text(f"Anomaly Score : {results['anomaly_score'].loc[results.index[0]]}")
            else:
                st.subheader("No Anomaly Detected!")

with tab2:
    st.header("Structure-based Anomaly Detection")
    st.write("Upload a PDB file to analyze structural anomalies")
    
    uploaded_file = st.file_uploader("Choose a PDB file", type="pdb")
    
    if uploaded_file is not None:
        # Read the PDB file
        pdb_content = uploaded_file.read()
        
        # Create a temporary file for PDB parser
        with open("temp.pdb", "wb") as f:
            f.write(pdb_content)
        
        # Parse the PDB file
        detector = ProteinAnomalyDetectorPDB()
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", "temp.pdb")
        
        if st.button("Detect Structural Anomalies"):
            with st.spinner("Analyzing structure..."):
                results = detector.detect_structure_anomalies(structure)
                
                anomalies = results['anomalies']
                structure_features = results['feature_vector']
                

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Anomalies", len(anomalies))
                with col2:
                    st.metric("Total Atoms", structure_features.get('atom_count', 'N/A'))
                with col3:
                    st.metric("Total Residues", structure_features.get('residue_count', 'N/A'))
                
                if anomalies:
                    anomaly_df = pd.DataFrame(anomalies)
                    
                    st.subheader("Anomaly Severity Distribution")
                    severity_counts = anomaly_df['severity'].value_counts()
                    fig_severity = px.pie(
                        names=severity_counts.index, 
                        values=severity_counts.values,
                        title="Anomaly Severity Distribution",
                        color_discrete_map={
                            'high': 'red', 
                            'medium': 'orange', 
                            'low': 'yellow'
                        }
                    )
                    st.plotly_chart(fig_severity)
                    
                    st.subheader("Detailed Anomaly Findings")
                    
                    with st.expander("View Detailed Anomalies"):
                        display_df = anomaly_df.copy()
                        
                        display_df['description'] = display_df.apply(lambda row: format_anomaly_description(row), axis=1)
                        
                        display_columns = ['type', 'severity', 'description']
                        st.dataframe(display_df[display_columns], use_container_width=True)
                    
                    st.subheader("Structural Metrics")
                    
                    # Create bar chart of key structural features
                    structural_metrics = {
                        'Atom Count': structure_features.get('atom_count', 0),
                        'Residue Count': structure_features.get('residue_count', 0),
                        'CA Distance Mean': structure_features.get('ca_distance_mean', 0),
                        'CA Distance Std': structure_features.get('ca_distance_std', 0),
                        'CA Max Distance': structure_features.get('ca_distance_max', 0)
                    }
                    
                    fig_metrics = go.Figure(data=[
                        go.Bar(
                            x=list(structural_metrics.keys()), 
                            y=list(structural_metrics.values()),
                            marker_color=['blue', 'green', 'red', 'purple', 'orange']
                        )
                    ])
                    fig_metrics.update_layout(
                        title='Key Structural Metrics',
                        xaxis_title='Metric',
                        yaxis_title='Value'
                    )
                    st.plotly_chart(fig_metrics)
                else:
                    st.success("No significant anomalies detected in the protein structure.")

with tab3:
    st.header("Machine Learning Anomaly Detection")
    st.write("Compare your protein to a reference set using machine learning")
    
    # Input protein
    input_method = st.radio("Select input method for target protein:", ["UniProt ID", "Paste sequence"], key="ml_input")
    
    target_sequence = None
    if input_method == "UniProt ID":
        uniprot_id = st.text_input("Enter UniProt ID:", "P01308", key="ml_uniprot")
        if st.button("Fetch Target Sequence"):
            with st.spinner("Fetching sequence..."):
                target_sequence = fetch_uniprot_data(uniprot_id)
                if target_sequence:
                    st.success(f"Retrieved sequence of length {len(target_sequence)}")
                else:
                    st.error("Failed to retrieve sequence")
    else:
        target_sequence = st.text_area("Enter protein sequence:", key="ml_sequence")
    
    # Reference proteins
    st.subheader("Reference Proteins")
    reference_method = st.radio("Select reference proteins:", ["Fetch by keyword", "Use built-in references"])
    
    reference_sequences = {}
    if reference_method == "Fetch by keyword":
        keyword = st.text_input("Enter keyword (e.g., 'insulin' or 'hemoglobin'):")
        limit = st.slider("Number of reference proteins:", 10, 50, 20)
        
        if st.button("Fetch Reference Proteins"):
            with st.spinner("Fetching reference proteins..."):
                reference_sequences = fetch_reference_proteins(keyword, limit)
                if reference_sequences:
                    st.success(f"Retrieved {len(reference_sequences)} reference proteins")
                else:
                    st.error("Failed to retrieve reference proteins")
    else:
        st.info("Using built-in reference set")
        # This would be replaced with actual built-in references
        reference_sequences = {"Built-in": "PLACEHOLDER"}
    
    # Run ML anomaly detection
    if target_sequence and reference_sequences:
        detector = ProteinAnomalyDetector()
        
        if st.button("Run ML Anomaly Detection"):
            with st.spinner("Analyzing with machine learning..."):
                results = detector.run_ml_anomaly_detection(target_sequence, reference_sequences)
                
                if 'error' in results:
                    st.error(results['error'])
                elif 'message' in results:
                    st.warning(results['message'])
                else:
                    # Display results
                    st.subheader("ML Anomaly Detection Results")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Anomaly Score", f"{results['anomaly_score']:.4f}")
                    with col2:
                        st.metric("Threshold", f"{results['threshold']:.4f}")
                    
                    if results['is_anomaly']:
                        st.warning("This protein is classified as an anomaly")
                    else:
                        st.success("This protein is within normal parameters")
                    
                    # Create a gauge chart
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = results['anomaly_score'],
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Anomaly Score"},
                        gauge = {
                            'axis': {'range': [None, 0]},
                            'steps': [
                                {'range': [results['threshold'], 0], 'color': "lightgray"},
                                {'range': [results['anomaly_score'], results['threshold']], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': results['threshold']
                            }
                        }
                    ))
                    
                    st.plotly_chart(fig)
