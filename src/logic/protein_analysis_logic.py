import json
import os
import re
import numpy as np
import pandas as pd
import requests
import streamlit as st
from google import genai
from google.oauth2 import service_account
from urllib.parse import quote
import py3Dmol
from Bio import PDB
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.distance import (
    add_aromatic_interactions,
    add_disulfide_interactions,
    add_hydrophobic_interactions,
    add_peptide_bonds,
)
from graphein.protein.visualisation import plotly_protein_structure_graph
from graphein.protein.analysis import plot_edge_type_distribution
from graphein.protein.analysis import plot_degree_by_residue_type
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
project = os.getenv("PROJECT_ID")

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service-cred.json'
# creds = service_account.Credentials.from_service_account_file("service-cred.json")

ai_client = genai.Client(api_key=api_key)

protein_cache = {}

def fetch_proteins_from_uniprot(query, max_results=10):
    try:
        encoded_query = quote(query)
        url = f"https://rest.uniprot.org/uniprotkb/search?query={encoded_query}&format=json&size={max_results}"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            results = []
            for item in data.get('results', []):
                protein_id = item.get('primaryAccession', '')
                protein_name = item.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
                if not protein_name:
                    protein_name = item.get('proteinDescription', {}).get('submissionNames', [{}])[0].get('fullName', {}).get('value', '')
                
                sequence = ''
                if 'sequence' in item:
                    sequence = item['sequence'].get('value', '')
                
                pfam_domains = []
                for feature in item.get('features', []):
                    if feature.get('type') == 'Domain' and 'description' in feature:
                        desc = feature.get('description')
                        if 'Pfam' in desc:
                            pfam_domains.append(desc)
                
                results.append({
                    'protein_id': protein_id,
                    'protein_name': protein_name,
                    'sequence': sequence,
                    'pfam': "; ".join(pfam_domains) if pfam_domains else "Not available",
                    'organism': item.get('organism', {}).get('scientificName', 'Unknown')
                })
            
            return pd.DataFrame(results)
        else:
            st.error(f"Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching from UniProt: {e}")
        return None


def fetch_proteins_from_organism(organism_id, max_results=10):
    try:
        url = f"https://rest.uniprot.org/uniprotkb/search?query=organism_id:{organism_id}&format=json"
        headers = {'Accept': 'application/json'}
        
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            
            results = []
            counter = 0
            

            for protein in data.get("results", []):
                protein_id = protein.get("primaryAccession")
                if counter >= max_results:
                    break
                
                uniprot_url = f"https://rest.uniprot.org/uniprotkb/{protein_id}"
                uniprot_response = requests.get(uniprot_url, headers={'Accept': 'application/json'})
                
                if uniprot_response.status_code == 200:
                    uniprot_data = uniprot_response.json()
                    
                    protein_name = uniprot_data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
                    sequence = uniprot_data.get('sequence', {}).get('value', '')
                    organism = uniprot_data.get('organism', {}).get('scientificName', 'Unknown')
                    
                    results.append({
                        'protein_id': protein_id,
                        'protein_name': protein_name,
                        'sequence': sequence,
                        'organism': organism
                    })
                    
                    counter += 1
            return pd.DataFrame(results)
        else:
            st.error(f"Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching from organism: {e}")
        
        #fallback
        return get_sample_data("PF07224")
    
def fetch_proteins_from_uniprot_organism(organism_id, query, max_results=10):
    try:
        encoded_query = quote(query)
        # organism_id = str(organism_id)
        # search_query = f"(organism_id:{organism_id}) AND ({query})"
        # params = {
        #     "query": search_query,
        #     "size": max_results,
        #     "format": "json"
        # }
        # headers = {
        #     "accept": "application/json"
        # }
        # base_url = "https://rest.uniprot.org/uniprotkb/search"

        # response = requests.get(base_url, headers=headers, params=params)

        url = f"https://rest.uniprot.org/uniprotkb/search?query=organism_id:{organism_id}%20{encoded_query}&format=json&size={max_results}"
        print(url)
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            results = []
            for item in data.get('results', []):
                protein_id = item.get('primaryAccession', '')
                protein_name = item.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', '')
                if not protein_name:
                    protein_name = item.get('proteinDescription', {}).get('submissionNames', [{}])[0].get('fullName', {}).get('value', '')
                
                sequence = ''
                if 'sequence' in item:
                    sequence = item['sequence'].get('value', '')
                
                
                results.append({
                    'protein_id': protein_id,
                    'protein_name': protein_name,
                    'sequence': sequence,
                    'organism': item.get('organism', {}).get('scientificName', 'Unknown')
                })
            
            return pd.DataFrame(results)
        else:
            st.error(f"Error: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error fetching from UniProt and organism: {e}")
        return None


def query_proteins(query_text, max_results=10):
    try:
        
        prompt = f"""
        Analyze this protein query and determine:
        1. What specific organism is being searched for (if any)
        2. What keyword terms should be used for searching
        3. What type of proteins the user is looking for
        
        Format your response as a JSON object with these fields:
        - organism_id: The Organism ID if present (like 3702), or null if not specified
        - search_terms: A list of search terms to use when querying protein databases
        - protein_type: Brief description of the type of proteins being searched for
        
        Query: {query_text}
        
        JSON:
        """
        
        response = ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        print("gemini response : ", response.text)
        
        try:
            json_text = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_text:
                query_params = json.loads(json_text.group())
            else:
                raise ValueError("Could not extract JSON from response")
                
            organism_id = query_params.get('organism_id')
            search_terms = query_params.get('search_terms', [])

            if organism_id and search_terms:
                search_query = " AND ".join(search_terms)
                return fetch_proteins_from_uniprot_organism(organism_id, search_query, max_results)
            elif organism_id:
                return fetch_proteins_from_organism(organism_id, max_results)
            elif search_terms:
                search_query = " AND ".join(search_terms)
                return fetch_proteins_from_uniprot(search_query, max_results)
            else:
                #default fallback
                return fetch_proteins_from_uniprot("hydrolase", max_results)
        except Exception as e:
            st.error(f"Error parsing Gemini response: {e}")
            search_terms = query_text.lower()
            organism_match = re.search(r'\b\d+\b', query_text)
            if organism_match:
                organism_id = organism_match.group()
                return fetch_proteins_from_organism(organism_id, max_results)
            
            return fetch_proteins_from_uniprot(query_text, max_results)
    except Exception as e:
        st.error(f"Error processing query: {e}")
        #fallback
        return get_sample_data()
    

# For Testing
def get_sample_data(pfam_id="PF07224"):
    sample_data = [
        {
            "protein_id": "A0A0K8P6T7",
            "protein_name": "Poly(ethylene terephthalate) hydrolase",
            "sequence": "MNFPRASRLMQAAVLGGLMAVSAAATAQTNPYARGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSSIKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNPSLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVAWMKRFMDNDTRYSTFACENPNSTRVSDFRTANCS",
            "pfam": f"{pfam_id}; Plasticized polyester hydrolase",
            "organism": "Ideonella sakaiensis"
        },
        {
            "protein_id": "A0A0K8P8E7",
            "protein_name": "MHETase",
            "sequence": "MATWKGLFPRSSKLAAIAAGLGFVCYSPHGIAAAAPAAYSSASQAAKLPAMATKQADIALGSDALAPNSNGSYSFATNTTRAFNSAKYGDPNNGYYGSWTQGLYTNAQFTSDNMPPMYNKSYAGDTLGGGAEFIFKNADRSTTFDANQVNNLQDYHYAGAAAAQNYFRISQFDDSYAAYESAQFEAQKSVYANSASLYATSDGTACLSRASDGANITFSHQDFLNHVPFYNSATGADSAVATYTKLSGTFSAPAALYTTTEDLPDSQGSGCPYYVNNHNLGDAGSKWTPPAPGCDDDSGLFTSTQQPGKWLTGATSCTFDGPAGAQVSYTDWSNTNANATLSASTASASGSSYGLGFASGDGAGAALALAPPPDPSNPPSPPFPAGCCTDSQEGTLVLTGNATATVSVTGLKKVGCAQGSPSHLSNQCNAWQGADCTDSGTLLGTAGTFQLVPTASAAGPGNPSLTGASVWELWKQNGWNCGNGLRSAILQSPALGTRGCSSTKYVSADSWCSSGSTMDDFNKLWFQQFGGSSDKRDSDDSSSCNWIRHSCSVNIQPAHSRLMNYASYWIMSNGDNSLTVDGNGTLLTIQNDFNPGSWHEVRLGHFRAGNADYDKASFSWSGNNYVDSYGTINHYGIYSHDSGLQKYPFNPYTTDPRQGDSMVIFSGDNNLIYGATGSSTFTVSLNASVDGQTDRNGGQWFWDAEDSPTNPNSPSTNGSVTVTQLPVSGAAGTMSYYTSGNDTTLQNKITIATIQASSDFPVCANSYTAVWDKPSGATFSFNYNQSFLSTSSAIDTSSWTAWATDCITNANGQTFTDQGKSYKCTFQDGSMAPGKTIVSINVTDNRTGVAAQTVTNQNGNYAVHSNASVTDSSQIRYNYKVDPSSSYVADQNTTLMADLKPSCCTYSSSSSCSA",
            "pfam": f"{pfam_id}; Terephthalate esterase",
            "organism": "Ideonella sakaiensis"
        },
        {
            "protein_id": "Q01684",
            "protein_name": "Cutinase",
            "sequence": "MLKQNFCRLLSSVAPLAAGLVSTPAAHAAIDPFTAETAYGAWDGKETATFPGSTSTCTTPTLVSTNMAKALAKQIHSAGAKHVFVFAQRLMNETARKGGAFPGFAPESCDKLIDCALAASKAATTWEGLVPSDALVATSGIDDAAHESQTFTSAIAAFVNKCPM",
            "pfam": f"{pfam_id}; Cutinase",
            "organism": "Fusarium solani"
        }
    ]
    return pd.DataFrame(sample_data)

def analyze_protein_with_gemini(protein_data):
    try:
        protein_id = protein_data.get('protein_id', '')
        protein_name = protein_data.get('protein_name', '')
        organism = protein_data.get('organism', '')
        sequence = protein_data.get('sequence', '')
        
        seq_length = len(sequence)
        
        prompt = f"""
        Analyze this protein:
        
        ID: {protein_id}
        Name: {protein_name}
        Organism: {organism}
        Sequence length: {seq_length} amino acids
        
        Please provide:
        1. A brief summary of the protein's likely function based on its domains
        2. Potential applications in biotechnology or medicine
        3. Key structural features that might be important
        4. Similar proteins or protein families
        5. Relevance to environmental processes or industrial applications
        6. If in context of organism, what possible effects can it have on the organism's health
        
        Format the analysis in clear sections with headers.
        Keep the analysis concise, scientific, and evidence-based.
        """
        
        response = ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"Error analyzing protein: {e}\n\nPlease try again or check your Gemini API configuration."
    
def display_protein_structure(pdb_data):
    view = py3Dmol.view(width=800, height=400)
    view.addModel(pdb_data, "pdb")
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.zoomTo()
    return view

def generate_protein_structure(sequence, protein_id):
    af_url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_id}-F1-model_v4.pdb"
    response = requests.get(af_url)
    if response.status_code == 200:
        # with open(f"{uniprot_id}.pdb", "wb") as f:
        #     f.write(response.content)
        # print(f"AlphaFold PDB saved as {uniprot_id}.pdb")
        pdb_data = response.text
        return pdb_data
    else:
        return None
    
#can be used, but needs pdb file 
def generate_visual_graphein(pdb_file):
    print("file : ", pdb_file)
    config = ProteinGraphConfig(
     edge_construction_functions=[       
         add_hydrophobic_interactions,
         add_aromatic_interactions,
         add_disulfide_interactions,
         add_peptide_bonds,
     ],
     #graph_metadata_functions=[asa, rsa],  # Add ASA and RSA features.
     #dssp_config=DSSPConfig(),             # Add DSSP config in order to compute ASA and RSA.
    )  
    print("after file")
    g = construct_graph(path="protein_graphs\protein.pdb", config=config)
    print("after graph")
    return g


def validate_pdb():
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", "protein_graphs\protein.pdb")
        print("✅ PDB file is valid and successfully parsed.")
    except Exception as e:
        st.error(f"PDB parsing error: {e}")
        print(f"❌ Error parsing PDB file: {e}")
