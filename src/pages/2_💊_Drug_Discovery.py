import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import Draw, Descriptors

import time

from logic.drug_discovery_logic import analyze_molecule, analyze_protein_disease_associations, calculate_druggability, fingerprint_to_smiles, generate_novel_compounds, get_alphafold_structure, get_compound_libraries, get_protein_info, get_real_compound_libraries, mol_to_svg, query_uniprot, screen_compounds, showmol
from models.DrugDiscoveryVAE.utils import create_fp, impl_pca, prep_data


st.set_page_config(
    page_title="Drug Discovery",
    page_icon="💊",
)

st.title("Drug Discovery")


with st.sidebar:
    st.header("About")
    st.info("""
    This app allows you to discover drugs for a proteins related to a given disease or condition. It analyzes the protein
    and determines its druggability score. For that particular protein, it performs screening and De-Novo design to generate drugs.
    
    Data sources: UniProt, Pfam
    """)
    
    st.header("Examples")
    st.markdown("""
    - Proteins related nerve degradation
    - Huntingtons causing proteins
    - Proteins that cause Alzheimer's 
    """)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Target Identification", "Target Analysis", "Virtual Screening", "De Novo Design", "Latent Space Exploration"])

with tab1:
    st.header("Target Identification")
    st.write("Search for proteins related to a specific disease or condition")
    
    query = st.text_input("Enter search query (e.g., 'Proteins related to nerve degradation')")
    
    if st.button("Search Protein Targets"):
        if query:
            with st.spinner("Searching for protein targets..."):
                results = query_uniprot(query)
                
                if results and "results" in results:
                    st.session_state.uniprot_results = results["results"]
                    
                    proteins_data = []
                    for protein in results["results"]:
                        protein_id = protein.get("primaryAccession", "N/A")
                        protein_name = protein.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
                        gene_name = protein.get("genes", [{}])[0].get("geneName", {}).get("value", "N/A") if protein.get("genes") else "N/A"
                        organism = protein.get("organism", {}).get("scientificName", "N/A")
                        
                        proteins_data.append({
                            "UniProt ID": protein_id,
                            "Protein Name": protein_name,
                            "Gene": gene_name,
                            "Organism": organism
                        })
                    
                    df = pd.DataFrame(proteins_data)
                    st.session_state.proteins_df = df
                    
                    st.write(f"Found {len(df)} proteins:")
                    st.dataframe(df)
                else:
                    st.error("No results found or error in API response")
        else:
            st.warning("Please enter a search query")

with tab2:
    st.header("Target Analysis")
    st.write("Analyze potential drug targets in detail")
    
    if "proteins_df" in st.session_state and "uniprot_results" in st.session_state:
        model_options = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash-preview-04-17"]
        selected_model = st.selectbox("Select a Google AI model for disease analysis", model_options)

        protein_options = st.session_state.proteins_df["UniProt ID"].tolist()
        selected_protein = st.selectbox("Select a protein to analyze", protein_options)
        
        if st.button("Analyze Selected Protein"):
            with st.spinner("Analyzing protein..."):
                protein_info = get_protein_info(selected_protein)
                
                if protein_info:
                    structure_data = get_alphafold_structure(selected_protein)
                    
                    if structure_data:
                        st.session_state.current_protein = {
                            "id": selected_protein,
                            "info": protein_info,
                            "structure": structure_data
                        }
                        
                        druggability_score = calculate_druggability(protein_info, structure_data)
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.subheader("Protein Information")
                            protein_name = protein_info.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
                            gene_name = protein_info.get("genes", [{}])[0].get("geneName", {}).get("value", "N/A") if protein_info.get("genes") else "N/A"
                            organism = protein_info.get("organism", {}).get("scientificName", "N/A")
                            
                            st.markdown(f"**Protein Name:** {protein_name}")
                            st.markdown(f"**Gene:** {gene_name}")
                            st.markdown(f"**Organism:** {organism}")
                            st.markdown(f"**UniProt ID:** {selected_protein}")
                            
                            st.subheader("Druggability Assessment")
                            st.markdown(f"**Druggability Score:** {druggability_score:.1f}/10")
                            
                            # Create a gauge-like visualization
                            fig, ax = plt.subplots(figsize=(10, 2))
                            ax.barh(0, druggability_score, color='blue', height=0.5)
                            ax.barh(0, 10, color='lightgray', height=0.5, alpha=0.3)
                            ax.set_xlim(0, 10)
                            ax.set_yticks([])
                            ax.set_xticks([0, 2, 4, 6, 8, 10])
                            ax.axvline(x=3.33, color='red', linestyle='--', alpha=0.7)
                            ax.axvline(x=6.67, color='red', linestyle='--', alpha=0.7)
                            ax.text(1.67, 0.25, "Low", ha='center')
                            ax.text(5, 0.25, "Medium", ha='center')
                            ax.text(8.33, 0.25, "High", ha='center')
                            st.pyplot(fig)
                            
                            if druggability_score < 3.33:
                                st.warning("This protein has low druggability potential.")
                            elif druggability_score < 6.67:
                                st.info("This protein has moderate druggability potential.")
                            else:
                                st.success("This protein has high druggability potential!")
                        
                        with col2:
                            st.subheader("Protein Structure (AlphaFold)")
                            mol_html = showmol(structure_data)
                            st.components.v1.html(mol_html, height=500)
                        
                        st.subheader("Disease Associations & Drug Target Analysis")
                        
                        analysis = analyze_protein_disease_associations(protein_info, model_name=selected_model)
                        st.markdown(analysis)
                        
                    else:
                        st.error("Could not retrieve AlphaFold structure")
                else:
                    st.error("Could not retrieve detailed protein information")
    else:
        st.info("Please search for proteins in the Target Identification tab first")

with tab3:
    st.header("Virtual Screening")
    st.write("Screen compound libraries against selected protein target")
    
    if "current_protein" in st.session_state:
        libraries = get_real_compound_libraries()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Select Compound Library")
            selected_library = st.selectbox("Choose a library", list(libraries.keys()))
            
            if st.button("Run Virtual Screening"):
                with st.spinner("Screening compounds..."):
                    # Get compounds from selected library
                    compounds = libraries[selected_library]
                    
                    # Run virtual screening
                    results = screen_compounds(st.session_state.current_protein["structure"], compounds)
                    
                    # Save results to session state
                    st.session_state.screening_results = results
                    
                    # Display results
                    st.success(f"Screened {len(results)} compounds successfully!")
        
        with col2:
            if "screening_results" in st.session_state:
                st.subheader("Screening Results")
                
                # Convert results to dataframe
                results_df = pd.DataFrame(st.session_state.screening_results)
                
                # Sort by binding affinity
                results_df = results_df.sort_values("Binding Affinity")
                
                # Display top results
                st.dataframe(results_df)
                
                # Plot binding affinities
                st.subheader("Binding Affinity Distribution")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(results_df["Binding Affinity"], kde=True, ax=ax)
                ax.set_xlabel("Binding Affinity (kcal/mol)")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)

                top_compounds = results_df.head(3)["SMILES"].tolist()
                
                mols = [Chem.MolFromSmiles(smiles) for smiles in top_compounds]
                
                st.subheader("Top 3 Compounds (SVG Rendering)")
                for i, mol in enumerate(mols):
                    svg = mol_to_svg(mol, legend=f"Score: {results_df.iloc[i]['Binding Affinity']:.2f}")
                    st.image(svg, use_container_width=False)
    else:
        st.info("Please select and analyze a protein target first")

with tab4:
    st.header("De Novo Drug Design")
    st.write("Generate novel compounds for the selected target")
    
    if "current_protein" in st.session_state:
        protein_id = st.session_state.current_protein["id"]
        protein_info = st.session_state.current_protein["info"]
        protein_name = protein_info.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
        
        st.subheader(f"Designing Novel Compounds for {protein_name}")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            num_compounds = st.slider("Number of compounds to generate", 1, 10, 5)
            
            if st.button("Generate Novel Compounds"):
                with st.spinner("Generating compounds..."):
                    time.sleep(2)
                    
                    compounds = generate_novel_compounds(protein_id, num_compounds)
                    
                    st.session_state.generated_compounds = compounds
                    
                    st.success(f"Generated {len(compounds)} novel compounds!")
        
        with col2:
            if "generated_compounds" in st.session_state:
                st.subheader("Generated Compounds")
                
                mols = [Chem.MolFromSmiles(smiles) for smiles in st.session_state.generated_compounds]

                st.subheader("Generated Compounds (SVG Rendering)")
                for i, mol in enumerate(mols):
                    svg = mol_to_svg(mol, legend=f"Compound {i+1}")
                    st.image(svg, use_container_width=False)
                
                st.subheader("Compound Properties")
                
                properties = []
                for i, mol in enumerate(mols):
                    props = {
                        "Compound": f"Compound {i+1}",
                        "SMILES": st.session_state.generated_compounds[i],
                        "Molecular Weight": Descriptors.MolWt(mol),
                        "LogP": Descriptors.MolLogP(mol),
                        "H-Bond Donors": Descriptors.NumHDonors(mol),
                        "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
                        "Lipinski Violations": Lipinski.NumRotatableBonds(mol) > 10 or
                                                Descriptors.MolWt(mol) > 500 or
                                                Descriptors.MolLogP(mol) > 5 or
                                                Descriptors.NumHDonors(mol) > 5 or
                                                Descriptors.NumHAcceptors(mol) > 10
                    }
                    properties.append(props)
                
                # Convert to dataframe
                props_df = pd.DataFrame(properties)
                st.dataframe(props_df)
                
                if st.button("Dock Generated Compounds"):
                    with st.spinner("Docking compounds..."):
                        screening_results = screen_compounds(st.session_state.current_protein["structure"], st.session_state.generated_compounds)
                        
                        results_df = pd.DataFrame(screening_results)
                        
                        results_df = results_df.sort_values("Binding Affinity")
                        
                        st.subheader("Docking Results")
                        st.dataframe(results_df)
                        
                        st.subheader("Binding Affinity Comparison")
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.barplot(x="SMILES", y="Binding Affinity", data=results_df, ax=ax)
                        ax.set_xticklabels([f"Cpd {i+1}" for i in range(len(results_df))])
                        ax.set_ylabel("Binding Affinity (kcal/mol)")
                        ax.set_xlabel("Compound")
                        st.pyplot(fig)
    else:
        st.info("Please select and analyze a protein target first")

with tab5:
    st.header("🧬 Latent Space Explorer for Drug Discovery")

    st.markdown("Visualize and interact with molecules in the latent space of a VAE trained on bioactivity (pIC50).")

    st.subheader("Latent Space Colored by pIC50")

    df, X, y, z, z_np = prep_data()
    z_pca = impl_pca()

    fig, ax = plt.subplots()
    scatter = ax.scatter(z_pca[:, 0], z_pca[:, 1], c=y.cpu(), cmap='viridis', s=10)
    fig.colorbar(scatter, label="pIC50")
    st.pyplot(fig)

    index = st.slider("Select Molecule Index", 0, len(df) - 1, 0)
    smiles = df.iloc[index]["smiles"]
    st.write(f"**Selected SMILES**: {smiles}")

    mol = Chem.MolFromSmiles(smiles)
    st.image(Draw.MolToImage(mol), caption="Selected Molecule")
    
    st.subheader("Molecular Properties")
    analyze_molecule(smiles)

    st.subheader("Generate New Molecule Around Latent Vector")

    if st.button("Generate"):
        new_fp = create_fp(index)

        new_smi = fingerprint_to_smiles(new_fp)
        st.write(f"**Generated SMILES**: {new_smi}")
        new_mol = Chem.MolFromSmiles(new_smi)
        st.image(Draw.MolToImage(new_mol), caption="Generated Molecule")
        analyze_molecule(new_smi)