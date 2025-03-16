# import os
# from dotenv import load_dotenv
# import streamlit as st
# from google.cloud import bigquery
# from google import genai

# from utils import nl_to_sql_with_gemini, run_bigquery

# # api_key = os.getenv("GEMINI_API_KEY")
# # project = os.getenv("PROJECT_ID")

# # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service-cred.json'

# # ai_client = genai.Client(api_key=api_key)
# # client = bigquery.Client()

# # st.title("🔬 Proteios-Mini, your AI powered drug discovery companion!")
# # st.write("Enter a disease or drug name to retrieve research insights")

# # query = st.text_input("Enter you query......")


# # if st.button("Generate insights") and query:
# #     with st.spinner("Fetching results...."):
# #         query_text = f"""
# #         SELECT 
# #             uniprotId, 
# #             uniprotDescription,
# #             organismScientificName, 
# #         FROM 
# #             `bigquery-public-data.deepmind_alphafold.metadata`
# #         WHERE 
# #             uniprotId LIKE '%{query}%'
# #         """

# #         df = client.query(query_text, project=project).to_dataframe()
# #         print("df", df)

# #         text = " ".join(df["organismScientificName"].tolist())
# #         print("contents", text)


# #     response = ""

# #     with st.spinner("Generating insights with AI..."):
# #         response = ai_client.models.generate_content(model="gemini-2.0-flash", contents=text)
# #         st.success("Insights Generated!!")

# #     st.subheader("Research Papers")
# #     for _, row in df.iterrows():
# #         st.markdown(f"**{row['uniprotId']}**")
# #         st.write(row["organismScientificName"])
# #         st.divider()

# #     st.subheader("AI Generated Insights")
# #     st.write(response.text)



# # Streamlit UI
# st.title("Protein Analysis App")
# st.subheader("Using EBI-MGnify Public Dataset")

# query_input = st.text_area("Enter your protein query:", 
#                            "Find the proteins that have the Pfam domain PF07224, which is related to plastic degradation.",
#                            height=100)

# if st.button("Search"):
#     with st.spinner("Processing query..."):
#         # Convert natural language to SQL using Gemini
#         sql_query = nl_to_sql_with_gemini(query_input)
        
#         # Show SQL (optional - for debugging)
#         with st.expander("View SQL Query"):
#             st.code(sql_query)
        
#         # Execute query
#         results = run_bigquery(sql_query)
        
#         if results is not None and not results.empty:
#             st.success(f"Found {len(results)} results")
            
#             # Display results table
#             st.dataframe(results)
            
#             # Add view protein structure option
#             if 'sequence' in results.columns:
#                 selected_protein = st.selectbox(
#                     "Select a protein to view its structure:",
#                     results['mgyp'].tolist()
#                 )
                
#                 if st.button("Generate Protein Structure"):
#                     with st.spinner("Predicting protein structure... This may take a few minutes."):
#                         # Get sequence for selected protein
#                         sequence = results[results['mgyp'] == selected_protein]['sequence'].iloc[0]
                        
#                         # Get protein info for display
#                         st.subheader(f"Protein: {selected_protein}")
                        
#                         # Display sequence
#                         with st.expander("Amino Acid Sequence"):
#                             st.text(sequence)
                        
#                         # Generate structure
#                         # pdb_data = generate_protein_structure(sequence)
                        
#                         # Display structure
#                         # st.subheader("Predicted 3D Structure")
#                         # structure_html = display_protein_structure(pdb_data)
#                         # st.components.v1.html(structure_html, height=500)
                        
#                         # Generate protein insights with Gemini
#                         with st.spinner("Generating protein insights..."):
#                             try:
#                                 gemini_model = genai.GenerativeModel('gemini-pro')
#                                 protein_info = results[results['mgyp'] == selected_protein].iloc[0]
                                
#                                 # Extract Pfam domains and other annotations
#                                 pfam_info = protein_info.get('pfam', 'Not available')
                                
#                                 prompt = f"""
#                                 Analyze this protein with ID {selected_protein}.
                                
#                                 Pfam domains: {pfam_info}
                                
#                                 Please provide:
#                                 1. A brief summary of the protein's likely function based on its domains
#                                 2. Potential applications in biotechnology or medicine
#                                 3. Similar proteins or protein families
                                
#                                 Keep the analysis concise and scientific.
#                                 """
                                
#                                 response = gemini_model.generate_content(prompt)
                                
#                                 with st.expander("Protein Analysis"):
#                                     st.markdown(response.text)
#                             except Exception as e:
#                                 st.error(f"Error generating insights: {e}")
                            
#                         # Download option
#                         # tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdb')
#                         # tmp.write(pdb_data.encode())
#                         # tmp_name = tmp.name
#                         # tmp.close()
                        
#                         # with open(tmp_name, "rb") as f:
#                         #     pdb_bytes = f.read()
#                         # os.unlink(tmp_name)
                        
#                         # b64 = base64.b64encode(pdb_bytes).decode()
#                         # href = f'<a href="data:file/pdb;base64,{b64}" download="{selected_protein}.pdb">Download PDB File</a>'
#                         # st.markdown(href, unsafe_allow_html=True)
#         else:
#             st.warning("No results found. Try modifying your query.")

# # Add sidebar with additional options
# with st.sidebar:
#     st.header("About")
#     st.markdown("""
#     This app provides access to protein data from the EBI-MGnify public dataset in BigQuery.
    
#     Features:
#     - Natural language querying using Gemini
#     - Protein structure prediction
#     - Interactive 3D visualization
#     - Protein function analysis
    
#     Data source: [EBI-MGnify](https://www.ebi.ac.uk/metagenomics/)
#     """)
    
#     st.header("Example Queries")
#     st.markdown("""
#     - Find proteins with the Pfam domain PF07224
#     - Show proteins related to plastic degradation
#     - Find proteins with hydrolase activity
#     - Show proteins from thermophilic organisms
#     """)