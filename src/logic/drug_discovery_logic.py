import py3Dmol
import streamlit as st
import numpy as np
import requests
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from google import genai 
from rdkit.Chem import rdMolDraw2D
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Lipinski


api_key = os.getenv("GEMINI_API_KEY")
project = os.getenv("PROJECT_ID")

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'service-cred.json'
# creds = service_account.Credentials.from_service_account_file("service-cred.json")

ai_client = genai.Client(api_key=api_key)

# #loadin model from cloud
# url = "https://s3.tebi.io/proteios-models/molecular_vae.pth"
# model_path = "molecular_vae.pth"
# if not os.path.exists(model_path):
#     response = requests.get(url)
#     with open(model_path, "wb") as f:
#         f.write(response.content)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# vae_model = MolecularVAE()
# vae_model.load_state_dict(torch.load(model_path, map_location=device))
# vae_model.to(device)

class MockMolecularDocking:
    def dock(self, receptor_file, ligand_file):
        # Simulate docking score
        return -8.5 + np.random.normal(0, 1)

class MockMoleculeGenerator:
    def generate(self, target_protein, n=5):
        smiles_list = [
            "CC(=O)NC1=CC=C(C=C1)O",
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
            "C1=CC=C2C(=C1)C=CC=C2OCCN3CCOCC3",
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5",
            "COC1=C(C=C2C(=C1)C(=NC(=N2)N3CCN(CC3)C)C4=CC=CC=C4)OCCCN5CCOCC5"
        ]
        return smiles_list

molecular_docking = MockMolecularDocking()
molecule_generator = MockMoleculeGenerator()


def get_real_compound_libraries():
    libraries = {}
    
    def get_drugbank_compounds():
        auth_header = {
            "Authorization": f"Bearer {os.environ.get('DRUGBANK_API_KEY')}"
        }
        response = requests.get(
            "https://api.drugbank.com/v1/drugs?approved=true&limit=100",
            headers=auth_header
        )
        if response.status_code == 200:
            drugs = response.json()
            return [drug["smiles"] for drug in drugs if "smiles" in drug]
        return [
            "CC(=O)NC1=CC=C(C=C1)O",  # Acetaminophen
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CCN(CC)CCOC1=CC=C(C=C1)C(=O)CCCC1=CC=C(C=C1)OC",  # Fexofenadine
            "CC(C)C1=C(C(=CC=C1)C(C)C)O",  # Thymol
        ]
    
    def get_chembl_compounds(query_type):
        base_url = "https://www.ebi.ac.uk/chembl/api/data"
        
        if query_type == "kinase_inhibitors":
            # Get kinase inhibitors
            response = requests.get(
                f"{base_url}/molecule?molecule_properties__alogp__lte=5&molecule_properties__full_mwt__lte=500" +
                "&target_type=SINGLE PROTEIN&target_component__accession=P00533" +  # EGFR kinase
                "&format=json&limit=100"
            )
        elif query_type == "fragments":
            # Get fragment-like compounds
            response = requests.get(
                f"{base_url}/molecule?molecule_properties__heavy_atoms__lte=20" +
                "&molecule_properties__alogp__lte=3&molecule_properties__full_mwt__lte=250" +
                "&format=json&limit=100"
            )
        
        if response.status_code == 200:
            results = response.json()
            compounds = []
            for molecule in results["molecules"]:
                if "molecule_structures" in molecule and "canonical_smiles" in molecule["molecule_structures"]:
                    compounds.append(molecule["molecule_structures"]["canonical_smiles"])
            return compounds
        return []
    
    def get_pubchem_compounds(query):
        search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/cids/JSON?name_type=word"
        response = requests.get(search_url)
        
        if response.status_code == 200:
            data = response.json()
            if "IdentifierList" in data and "CID" in data["IdentifierList"]:
                cids = data["IdentifierList"]["CID"][:100]
                
                cids_str = ",".join(map(str, cids))
                props_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids_str}/property/CanonicalSMILES/JSON"
                props_response = requests.get(props_url)
                
                if props_response.status_code == 200:
                    props_data = props_response.json()
                    if "PropertyTable" in props_data and "Properties" in props_data["PropertyTable"]:
                        return [prop["CanonicalSMILES"] for prop in props_data["PropertyTable"]["Properties"]]
        return []
    
    try:
        libraries["FDA-approved drugs"] = get_drugbank_compounds()
    except Exception as e:
        print("drugbank error : ", e)
        st.warning(f"Could not fetch DrugBank compounds: {str(e)}")
        libraries["FDA-approved drugs"] = [
            "CC(=O)NC1=CC=C(C=C1)O",  # Acetaminophen
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CCN(CC)CCOC1=CC=C(C=C1)C(=O)CCCC1=CC=C(C=C1)OC",  # Fexofenadine
            "CC(C)C1=C(C(=CC=C1)C(C)C)O",  # Thymol
        ]
        
    try:
        libraries["Kinase inhibitors"] = get_chembl_compounds("kinase_inhibitors")
    except Exception as e:
        print("chembl error : ", e)
        st.warning(f"Could not fetch ChEMBL kinase inhibitors: {str(e)}")
        libraries["Kinase inhibitors"] = []
        
    try:
        libraries["Natural products"] = get_pubchem_compounds("natural")
    except Exception as e:
        print("pubchem error : ", e)
        st.warning(f"Could not fetch PubChem natural products: {str(e)}")
        libraries["Natural products"] = []

    return libraries

def query_uniprot(query_term):
    """Query UniProt API for proteins based on search term"""
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": query_term,
        "format": "json",
        "size": 10
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def get_protein_info(uniprot_id):
    """Get detailed protein information from UniProt"""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}"
    response = requests.get(url, params={"format": "json"})
    if response.status_code == 200:
        return response.json()
    return None

def get_alphafold_structure(uniprot_id):
    """Download AlphaFold structure from AlphaFold DB"""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.pdb"
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    return None

def calculate_druggability(protein_info, structure):
    """Calculate druggability score for the protein"""
    score = 0
    
    # Check if the protein has known disease associations
    if protein_info.get("comments"):
        for comment in protein_info.get("comments", []):
            if comment.get("commentType") == "DISEASE":
                score += 2
    
    # Check if it's a kinase, GPCR, ion channel, or other druggable class
    protein_name = protein_info.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
    druggable_terms = ["kinase", "receptor", "channel", "reductase", "protease", "phosphatase"]
    if any(term in protein_name.lower() for term in druggable_terms):
        score += 1.5
    
    # Check if it's a membrane protein (often good drug targets)
    if protein_info.get("subcellularLocations"):
        for location in protein_info.get("subcellularLocations", []):
            if location.get("location") and "membrane" in str(location.get("location")).lower():
                score += 1
    
    # Add some random variation to simulate structural analysis
    score += np.random.uniform(0, 1)
    
    # Normalize to 0-10 scale
    return min(max(score, 0), 10)

def screen_compounds(target_protein_file, compound_library):
    """Screen compounds against target protein"""
    results = []
    
    for compound in compound_library:
        mol = Chem.MolFromSmiles(compound)
        if mol:
            mol_weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            h_donors = Descriptors.NumHDonors(mol)
            h_acceptors = Descriptors.NumHAcceptors(mol)
            
            # Calculate binding affinity using mock docking
            binding_affinity = molecular_docking.dock(target_protein_file, mol)
            
            # Calculate drug-likeness score
            drug_likeness = 0
            if 200 <= mol_weight <= 500:
                drug_likeness += 1
            if logp <= 5:
                drug_likeness += 1
            if h_donors <= 5:
                drug_likeness += 1
            if h_acceptors <= 10:
                drug_likeness += 1
            
            results.append({
                "SMILES": compound,
                "Molecular Weight": mol_weight,
                "LogP": logp,
                "H-Bond Donors": h_donors,
                "H-Bond Acceptors": h_acceptors,
                "Binding Affinity": binding_affinity,
                "Drug-likeness": drug_likeness
            })
    
    return results

def generate_novel_compounds(target_protein, n=5):
    """Generate novel compounds for target protein"""
    # In a real implementation, this would use a generative model
    # For now, we'll use our mock generator
    return molecule_generator.generate(target_protein, n)

def get_compound_libraries():
    """Return a list of available compound libraries"""
    #test-impl
    return {
        "FDA-approved drugs": [
            "CC(=O)NC1=CC=C(C=C1)O",  # Acetaminophen
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CCN(CC)CCOC1=CC=C(C=C1)C(=O)CCCC1=CC=C(C=C1)OC",  # Fexofenadine
            "CC(C)C1=C(C(=CC=C1)C(C)C)O",  # Thymol
        ],
        "Natural products": [
            "CC1C2C(C(=O)C3(C(CC4C(C3C(C(C2OC1=O)O)O)C(=O)OC4OC5CC(C(C(O5)C)O)(C)O)O)C)O",  # Erythromycin
            "CC1(C2CCC3(C(C2(CCC1O)C)CCC4=CC(=O)CCC34C)C)C",  # Testosterone
            "C1=CC=C2C(=C1)C(=CN2)CC(C(=O)O)N",  # Tryptophan
            "CC(C)C1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CC1C=CC2=C(C1=O)C3=C(C=C2)OC(=O)C3",  # Coumarin
        ],
        "Fragment library": [
            "C1=CC=NC=C1",  # Pyridine
            "C1=CC=CO1",  # Furan
            "C1=CSC=C1",  # Thiophene
            "C1=CN=CN1",  # Imidazole
            "C1CCCCC1",  # Cyclohexane
        ]
    }

def analyze_protein_disease_associations(protein_info):
    protein_name = protein_info.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
    gene_name = protein_info.get("genes", [{}])[0].get("geneName", {}).get("value", "")
    
    disease_info = []
    if protein_info.get("comments"):
        for comment in protein_info.get("comments", []):
            if comment.get("commentType") == "DISEASE":
                disease_info.append(comment.get("text", [{}])[0].get("value", ""))
    
    prompt = f"""
    Analyze the protein {protein_name} (gene: {gene_name}) and its potential as a drug target.
    
    Known disease associations:
    {"; ".join(disease_info) if disease_info else "None documented in UniProt"}
    
    Please provide:
    1. A brief summary of this protein's function
    2. Its potential as a drug target (high/medium/low)
    3. Known drug interactions or existing drugs that target it
    4. Potential mechanisms for drug targeting
    5. Any challenges or considerations for drug development
    
    Limit your response to 300 words.
    """
    
    try:
        response = ai_client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        return response.text
    except Exception as e:
        return f"Error in Gemini analysis: {str(e)}"


def showmol(structure_data, height=500, width=500):
    structure_viewer = py3Dmol.view(width=350, height=400)
    structure_viewer.addModel(structure_data.decode('utf-8'), 'pdb')
    structure_viewer.setStyle({'cartoon': {'color': 'spectrum'}})
    structure_viewer.zoomTo()
    structure_viewer.spin(True)
    html = structure_viewer._make_html()
    return html

def mol_to_svg(mol, legend=""):
    AllChem.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    drawer.DrawMolecule(mol, legend=legend)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()