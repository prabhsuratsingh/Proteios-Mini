import numpy as np
import pandas as pd
from Bio import SeqIO, AlignIO
from Bio.PDB import PDBParser
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import requests
import io
import plotly.graph_objects as go
import plotly.express as px

from models.AnomalyDetection.utils import load_model_project, prepare_protein_data

class ProteinAnomalyDetectorPDB:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
        self.aa_properties = {
            'A': {'hydrophobicity': 1.8, 'volume': 88.6, 'charge': 0},
            'R': {'hydrophobicity': -4.5, 'volume': 173.4, 'charge': 1},
            'N': {'hydrophobicity': -3.5, 'volume': 114.1, 'charge': 0},
            'D': {'hydrophobicity': -3.5, 'volume': 111.1, 'charge': -1},
            'C': {'hydrophobicity': 2.5, 'volume': 108.5, 'charge': 0},
            'Q': {'hydrophobicity': -3.5, 'volume': 143.8, 'charge': 0},
            'E': {'hydrophobicity': -3.5, 'volume': 138.4, 'charge': -1},
            'G': {'hydrophobicity': -0.4, 'volume': 60.1, 'charge': 0},
            'H': {'hydrophobicity': -3.2, 'volume': 153.2, 'charge': 0.1},
            'I': {'hydrophobicity': 4.5, 'volume': 166.7, 'charge': 0},
            'L': {'hydrophobicity': 3.8, 'volume': 166.7, 'charge': 0},
            'K': {'hydrophobicity': -3.9, 'volume': 168.6, 'charge': 1},
            'M': {'hydrophobicity': 1.9, 'volume': 162.9, 'charge': 0},
            'F': {'hydrophobicity': 2.8, 'volume': 189.9, 'charge': 0},
            'P': {'hydrophobicity': -1.6, 'volume': 112.7, 'charge': 0},
            'S': {'hydrophobicity': -0.8, 'volume': 89.0, 'charge': 0},
            'T': {'hydrophobicity': -0.7, 'volume': 116.1, 'charge': 0},
            'W': {'hydrophobicity': -0.9, 'volume': 227.8, 'charge': 0},
            'Y': {'hydrophobicity': -1.3, 'volume': 193.6, 'charge': 0},
            'V': {'hydrophobicity': 4.2, 'volume': 140.0, 'charge': 0}
        }
        self.reference_frequencies = {
            'A': 0.074, 'R': 0.042, 'N': 0.044, 'D': 0.059, 'C': 0.033,
            'Q': 0.037, 'E': 0.058, 'G': 0.074, 'H': 0.029, 'I': 0.038,
            'L': 0.076, 'K': 0.072, 'M': 0.018, 'F': 0.040, 'P': 0.050,
            'S': 0.081, 'T': 0.062, 'W': 0.013, 'Y': 0.033, 'V': 0.068
        }
        
    def calculate_sequence_features(self, sequence):
        """Calculate various sequence-based features for anomaly detection"""
        cleaned_seq = ''.join([aa for aa in sequence if aa in self.aa_properties])
        
        if not cleaned_seq:
            return None
            
        # Create ProtParam object
        try:
            analysis = ProteinAnalysis(cleaned_seq)
        except:
            # Handle sequences with non-standard amino acids
            return None
            
        # Calculate amino acid composition
        aa_composition = analysis.amino_acids_percent
        
        # Calculate basic properties
        features = {
            'length': len(cleaned_seq),
            'molecular_weight': analysis.molecular_weight(),
            'aromaticity': analysis.aromaticity(),
            'instability_index': analysis.instability_index(),
            'isoelectric_point': analysis.isoelectric_point(),
            'gravy': analysis.gravy(),
        }
        
        # Calculate amino acid frequencies
        aa_counts = {}
        for aa in self.aa_properties.keys():
            aa_counts[f'freq_{aa}'] = cleaned_seq.count(aa) / len(cleaned_seq) if len(cleaned_seq) > 0 else 0
            
        # Calculate dipeptide frequencies (for select dipeptides known to be important)
        important_dipeptides = ['GG', 'PP', 'CC', 'SS', 'LL', 'AA']
        for dipeptide in important_dipeptides:
            count = 0
            for i in range(len(cleaned_seq) - 1):
                if cleaned_seq[i:i+2] == dipeptide:
                    count += 1
            features[f'dipeptide_{dipeptide}'] = count / (len(cleaned_seq) - 1) if len(cleaned_seq) > 1 else 0
            
        # Calculate sequence complexity (Shannon entropy)
        aa_freq = {}
        for aa in cleaned_seq:
            aa_freq[aa] = aa_freq.get(aa, 0) + 1
        entropy = 0
        for aa, count in aa_freq.items():
            p = count / len(cleaned_seq)
            entropy -= p * np.log2(p)
        features['sequence_entropy'] = entropy
        
        # Combine all features
        features.update(aa_counts)
        
        return features
        
    def calculate_structure_features(self, pdb_structure):
        """Calculate structural features from a PDB structure"""
        features = {}
        
        # Get all atoms
        atoms = list(pdb_structure.get_atoms())
        
        # Calculate basic structure statistics
        features['atom_count'] = len(atoms)
        
        # Calculate residue-level statistics
        residues = list(pdb_structure.get_residues())
        features['residue_count'] = len(residues)
        
        # Calculate CA distances (as a proxy for protein compactness)
        ca_atoms = [atom for atom in atoms if atom.get_name() == 'CA']
        if len(ca_atoms) > 1:
            ca_coords = np.array([atom.get_coord() for atom in ca_atoms])
            # Calculate pairwise distances
            distances = []
            for i in range(len(ca_coords)):
                for j in range(i+1, len(ca_coords)):
                    distances.append(np.linalg.norm(ca_coords[i] - ca_coords[j]))
            
            # Calculate statistics on distances
            features['ca_distance_mean'] = np.mean(distances)
            features['ca_distance_std'] = np.std(distances)
            features['ca_distance_max'] = np.max(distances)
            features['ca_distance_min'] = np.min(distances)
        
        return features
        
    
    def detect_structure_anomalies(self, pdb_structure, reference_structures=None):
        structure_features = self.calculate_structure_features(pdb_structure)
        
        anomalies = []
        
        if reference_structures:
            ref_ca_distances = []
            ref_atom_counts = []
            ref_residue_counts = []
            
            for ref_structure in reference_structures:
                ref_features = self.calculate_structure_features(ref_structure)
                ref_ca_distances.append(ref_features.get('ca_distance_mean', 0))
                ref_atom_counts.append(ref_features.get('atom_count', 0))
                ref_residue_counts.append(ref_features.get('residue_count', 0))
            
            import numpy as np
            mean_ref_atoms = np.mean(ref_atom_counts)
            std_ref_atoms = np.std(ref_atom_counts)
            mean_ref_residues = np.mean(ref_residue_counts)
            std_ref_residues = np.std(ref_residue_counts)
            mean_ref_ca_distance = np.mean(ref_ca_distances)
            std_ref_ca_distance = np.std(ref_ca_distances)
        else:
            mean_ref_atoms = structure_features['atom_count']
            std_ref_atoms = mean_ref_atoms * 0.2
            mean_ref_residues = structure_features['residue_count']
            std_ref_residues = mean_ref_residues * 0.2
            mean_ref_ca_distance = structure_features.get('ca_distance_mean', 0)
            std_ref_ca_distance = mean_ref_ca_distance * 0.2
        
        z_score_atoms = (structure_features['atom_count'] - mean_ref_atoms) / std_ref_atoms if std_ref_atoms != 0 else 0
        if abs(z_score_atoms) > 2:
            anomalies.append({
                'type': 'abnormal_atom_count',
                'current_count': structure_features['atom_count'],
                'reference_mean': mean_ref_atoms,
                'z_score': z_score_atoms,
                'severity': 'high' if abs(z_score_atoms) > 3 else 'medium'
            })
        
        z_score_residues = (structure_features['residue_count'] - mean_ref_residues) / std_ref_residues if std_ref_residues != 0 else 0
        if abs(z_score_residues) > 2:
            anomalies.append({
                'type': 'abnormal_residue_count',
                'current_count': structure_features['residue_count'],
                'reference_mean': mean_ref_residues,
                'z_score': z_score_residues,
                'severity': 'high' if abs(z_score_residues) > 3 else 'medium'
            })
        
        if 'ca_distance_mean' in structure_features:
            z_score_ca_distance = (structure_features['ca_distance_mean'] - mean_ref_ca_distance) / std_ref_ca_distance if std_ref_ca_distance != 0 else 0
            
            if abs(z_score_ca_distance) > 2:
                anomalies.append({
                    'type': 'unusual_ca_distance',
                    'current_mean_distance': structure_features['ca_distance_mean'],
                    'reference_mean_distance': mean_ref_ca_distance,
                    'z_score': z_score_ca_distance,
                    'severity': 'high' if abs(z_score_ca_distance) > 3 else 'medium'
                })
            
            if structure_features['ca_distance_max'] > mean_ref_ca_distance + 3 * std_ref_ca_distance:
                anomalies.append({
                    'type': 'extreme_max_ca_distance',
                    'max_distance': structure_features['ca_distance_max'],
                    'reference_mean': mean_ref_ca_distance,
                    'severity': 'high'
                })
        
        backbone_atoms = ['N', 'CA', 'C', 'O']
        backbone_present = {atom: 0 for atom in backbone_atoms}
        
        for atom in pdb_structure.get_atoms():
            if atom.get_name() in backbone_atoms:
                backbone_present[atom.get_name()] += 1
        
        missing_backbone_atoms = [atom for atom, count in backbone_present.items() if count == 0]
        if missing_backbone_atoms:
            anomalies.append({
                'type': 'missing_backbone_atoms',
                'missing_atoms': missing_backbone_atoms,
                'severity': 'high'
            })
        
        print("Adaptive Anomalies:", anomalies)
        
        return {
            'feature_vector': structure_features,
            'anomalies': anomalies
        }

    def run_ml_anomaly_detection(self, sequence, reference_sequences):
        """Use machine learning to detect anomalies"""
        # Collect features for all sequences
        all_features = []
        all_ids = []
        
        # Process reference sequences
        for ref_id, ref_seq in reference_sequences.items():
            features = self.calculate_sequence_features(ref_seq)
            if features:
                all_features.append(features)
                all_ids.append(ref_id)
        
        # Process target sequence
        target_features = self.calculate_sequence_features(sequence)
        if not target_features:
            return {
                'error': 'Could not calculate features for the sequence',
                'anomaly_score': None
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # If we have enough reference sequences, perform ML anomaly detection
        if len(df) >= 10:
            # Select numerical columns
            num_cols = df.select_dtypes(include=[np.number]).columns
            
            # Standardize features
            scaler = StandardScaler()
            X = scaler.fit_transform(df[num_cols])
            
            # Apply Isolation Forest    
            clf = IsolationForest(contamination=0.01, random_state=42)
            clf.fit(X)
            
            # Transform target features
            # target_vector = scaler.transform(np.array([list(target_features[col] for col in num_cols)]))
            target_vector = pd.DataFrame([target_features], columns=num_cols)
            target_vector = scaler.transform(target_vector)
            
            train_scores = clf.decision_function(X)
            threshold = np.percentile(train_scores, 5)
            # anomaly_score = clf.score_samples(target_vector)[0]
            decision_score = clf.decision_function(target_vector)[0]
            is_anomaly = clf.predict(target_vector)[0] == -1

            print("Min score:", np.min(train_scores))
            print("Max score:", np.max(train_scores))
            print("10th percentile (threshold):", threshold)
            print("25th percentile:", np.percentile(train_scores, 25))
            print("50th percentile (median):", np.percentile(train_scores, 50))
            print("75th percentile:", np.percentile(train_scores, 75))
            print("Is anomaly : ", is_anomaly)
            # print(f"Anomaly Score: {anomaly_score}")  
            print(f"Decision Function Score: {decision_score}")

            return {
                'anomaly_score': decision_score,
                'is_anomaly': is_anomaly,
                'threshold': threshold,
                'severity': 'high' if is_anomaly else 'none'
            }
        else:
            return {
                'message': 'Not enough reference sequences for ML detection',
                'anomaly_score': None
            }
    
    def visualize_anomalies(self, sequence, anomalies):
        """Generate visualizations for detected anomalies"""
        # Create a figure to show amino acid composition anomalies
        aa_freq_anomalies = [a for a in anomalies if a['type'] == 'amino_acid_frequency']
        
        if aa_freq_anomalies:
            # Prepare data for plotting
            aas = []
            observed = []
            expected = []
            z_scores = []
            
            for anomaly in aa_freq_anomalies:
                aas.append(anomaly['amino_acid'])
                observed.append(anomaly['observed'])
                expected.append(anomaly['expected'])
                z_scores.append(anomaly['z_score'])
            
            # Create a DataFrame
            df = pd.DataFrame({
                'amino_acid': aas,
                'observed': observed,
                'expected': expected,
                'z_score': z_scores
            })
            
            # Create a bar plot using Plotly
            fig = px.bar(df, x='amino_acid', y=['observed', 'expected'], 
                         barmode='group', title='Anomalous Amino Acid Frequencies')
            
            # Add a line for z-scores
            fig_z = px.line(df, x='amino_acid', y='z_score', title='Z-Scores')
            
            # Combine the two plots
            for trace in fig_z.data:
                fig.add_trace(trace)
                
            return fig
        
        # Create a figure to show hydrophobicity profile
        hydrophobicity_profile = []
        for i in range(len(sequence)):
            aa = sequence[i]
            hydrophobicity = self.aa_properties.get(aa, {'hydrophobicity': 0})['hydrophobicity']
            hydrophobicity_profile.append(hydrophobicity)
        
        # Create a line plot
        fig = px.line(x=list(range(1, len(sequence) + 1)), y=hydrophobicity_profile,
                      title='Hydrophobicity Profile', labels={'x': 'Position', 'y': 'Hydrophobicity'})
        
        # Add markers for hydrophobic region anomalies
        hydrophobic_anomalies = [a for a in anomalies if a['type'] == 'hydrophobic_region']
        for anomaly in hydrophobic_anomalies:
            pos = anomaly['position'].split('-')
            if len(pos) == 2:
                start, end = int(pos[0]), int(pos[1])
                fig.add_shape(type="rect", x0=start, y0=-5, x1=end, y1=5,
                              line=dict(color="Red"), fillcolor="Red", opacity=0.2)
        
        return fig
    

def fetch_uniprot_data(accession):
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the FASTA content
        fasta_io = io.StringIO(response.text)
        records = list(SeqIO.parse(fasta_io, "fasta"))
        if records:
            return str(records[0].seq)
    return None

def fetch_reference_proteins(keyword, limit=20):
    url = f"https://rest.uniprot.org/uniprotkb/search?query={keyword}&format=fasta&size={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the FASTA content
        fasta_io = io.StringIO(response.text)
        records = list(SeqIO.parse(fasta_io, "fasta"))
        
        sequences = {}
        for record in records:
            sequences[record.id] = str(record.seq)
        
        return sequences
    return {}

def detect_anomaly_AE_model(sequence, id):
    model = load_model_project()
    new_sequences, new_ids = [sequence],[id]
    new_features = prepare_protein_data(new_sequences, feature_type="composition")
    new_anomalies, new_scores = model.detect_anomalies(new_features, return_scores=True)
    new_results = pd.DataFrame({
        'protein_id': new_ids,
        'anomaly_score': new_scores,
        'is_anomalous': new_anomalies,
        'seq': sequence
    })
    print("Analysis results for new proteins:")
    print(new_results.sort_values('anomaly_score', ascending=False))
    return new_results


def format_anomaly_description(anomaly):
    descriptions = {
        'abnormal_atom_count': f"Unusual total atom count. Current count differs significantly from expected (Z-score: {anomaly.get('z_score', 'N/A')})",
        'abnormal_residue_count': f"Unusual total residue count. Current count differs significantly from expected (Z-score: {anomaly.get('z_score', 'N/A')})",
        'unusual_ca_distance': f"Atypical alpha-carbon distances. Mean distance deviates from expected (Z-score: {anomaly.get('z_score', 'N/A')})",
        'extreme_max_ca_distance': f"Extremely large maximum alpha-carbon distance ({anomaly.get('max_distance', 'N/A')})",
        'missing_backbone_atoms': f"Critical structural issue: Missing backbone atoms ({', '.join(anomaly.get('missing_atoms', []))})",
        'missing_side_chain_atoms': f"Structural incompleteness: Missing side chain atoms ({', '.join(anomaly.get('missing_atoms', []))})",
        'unusual_hydrogen_distribution': f"Atypical hydrogen atom distribution (Ratio: {anomaly.get('hydrogen_ratio', 'N/A')})",
        'high_distance_variability': f"High variability in alpha-carbon distances (Std Dev: {anomaly.get('distance_std', 'N/A')})",
        'unusual_hydrophobic_balance': f"Unusual hydrophobic to polar atom balance (Ratio: {anomaly.get('hydrophobic_ratio', 'N/A')})"
    }
    
    return descriptions.get(anomaly['type'], f"Anomaly of type: {anomaly['type']}")