import streamlit as st
from logic.anomaly_detection import *

st.set_page_config(
    page_title="Anomaly Detection",
    page_icon="🧫",
)

st.title("Anomaly Detection")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Sequence Analysis", "Structure Analysis", "ML Anomaly Detection"])

with tab1:
    st.header("Sequence-based Anomaly Detection")
    
    # Input options
    input_method = st.radio("Select input method:", ["UniProt ID", "Paste sequence"])
    
    sequence = None
    if input_method == "UniProt ID":
        uniprot_id = st.text_input("Enter UniProt ID:", "P01308")
        if st.button("Fetch Sequence"):
            with st.spinner("Fetching sequence..."):
                sequence = fetch_uniprot_data(uniprot_id)
                if sequence:
                    st.success(f"Retrieved sequence of length {len(sequence)}")
                    st.code(sequence if len(sequence) <= 100 else sequence[:100] + "...")
                else:
                    st.error("Failed to retrieve sequence")
    else:
        sequence = st.text_area("Enter protein sequence:", "MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAEDLQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN")
    
    if sequence:
        # Run anomaly detection
        detector = ProteinAnomalyDetector()
        
        if st.button("Detect Anomalies"):
            with st.spinner("Analyzing sequence..."):
                results = detector.detect_sequence_anomalies(sequence)
                
                # Display results
                if 'error' in results:
                    st.error(results['error'])
                else:
                    anomalies = results['anomalies']
                    
                    if anomalies:
                        st.subheader(f"Found {len(anomalies)} anomalies")
                        
                        # Create a table of anomalies
                        anomaly_df = pd.DataFrame(anomalies)
                        st.dataframe(anomaly_df)
                        
                        # Visualize anomalies
                        fig = detector.visualize_anomalies(sequence, anomalies)
                        st.plotly_chart(fig)
                    else:
                        st.info("No anomalies detected in the sequence")

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
        detector = ProteinAnomalyDetector()
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", "temp.pdb")
        
        # Run anomaly detection
        if st.button("Detect Structural Anomalies"):
            with st.spinner("Analyzing structure..."):
                results = detector.detect_structure_anomalies(structure)
                
                # Display results
                anomalies = results['anomalies']
                
                if anomalies:
                    st.subheader(f"Found {len(anomalies)} structural anomalies")
                    
                    # Create a table of anomalies
                    anomaly_df = pd.DataFrame(anomalies)
                    st.dataframe(anomaly_df)
                else:
                    st.info("No structural anomalies detected")

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
