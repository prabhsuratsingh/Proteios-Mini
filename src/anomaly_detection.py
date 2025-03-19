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

class ProteinAnomalyDetector:
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
        # Remove non-standard amino acids
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
        aa_composition = analysis.get_amino_acids_percent()
        
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
        
    def detect_sequence_anomalies(self, sequence, reference_sequences=None):
        """Detect anomalies in protein sequences"""
        # Calculate features for the target sequence
        target_features = self.calculate_sequence_features(sequence)
        
        if not target_features:
            return {
                'error': 'Could not calculate features for the sequence',
                'anomalies': []
            }
        
        anomalies = []
        
        # Method 1: Compare amino acid frequencies with reference
        for aa, ref_freq in self.reference_frequencies.items():
            observed_freq = sequence.count(aa) / len(sequence)
            z_score = (observed_freq - ref_freq) / (np.sqrt(ref_freq * (1 - ref_freq) / len(sequence)))
            
            if abs(z_score) > 3:  # Threshold for statistical significance
                anomalies.append({
                    'type': 'amino_acid_frequency',
                    'amino_acid': aa,
                    'observed': observed_freq,
                    'expected': ref_freq,
                    'z_score': z_score,
                    'severity': 'high' if abs(z_score) > 5 else 'medium'
                })
        
        # Method 2: Check for unusual regions (sliding window approach)
        window_size = 10
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i+window_size]
            
            # Calculate hydrophobicity
            hydrophobicity = sum(self.aa_properties.get(aa, {'hydrophobicity': 0})['hydrophobicity'] for aa in window) / window_size
            
            # Detect hydrophobic regions in unexpected places
            if i < len(sequence) / 3:  # Signal peptide region
                if hydrophobicity > 2:
                    anomalies.append({
                        'type': 'hydrophobic_region',
                        'position': f"{i+1}-{i+window_size}",
                        'hydrophobicity': hydrophobicity,
                        'severity': 'medium'
                    })
            
            # Check for unusual charge distribution
            charge = sum(self.aa_properties.get(aa, {'charge': 0})['charge'] for aa in window)
            if abs(charge) > 5:  # Very high charge concentration
                anomalies.append({
                    'type': 'charge_cluster',
                    'position': f"{i+1}-{i+window_size}",
                    'charge': charge,
                    'severity': 'medium' if abs(charge) > 5 else 'low'
                })
        
        # Method 3: Check for unusual amino acid patterns
        patterns = {
            'poly_proline': 'PPP',
            'poly_glycine': 'GGG',
            'poly_glutamine': 'QQQ',  # Associated with some diseases
            'poly_alanine': 'AAA'
        }
        
        for pattern_name, pattern in patterns.items():
            positions = []
            pos = 0
            while True:
                pos = sequence.find(pattern, pos)
                if pos == -1:
                    break
                positions.append(pos + 1)  # 1-indexed positions
                pos += 1
            
            if positions:
                anomalies.append({
                    'type': 'unusual_pattern',
                    'pattern': pattern_name,
                    'positions': positions,
                    'severity': 'medium' if len(positions) > 1 else 'low'
                })
        
        # Method 4: Check for rare amino acids
        rare_aas = ['W', 'C', 'H', 'M']
        for aa in rare_aas:
            count = sequence.count(aa)
            freq = count / len(sequence)
            if freq > 2 * self.reference_frequencies[aa]:
                anomalies.append({
                    'type': 'high_rare_aa',
                    'amino_acid': aa,
                    'count': count,
                    'frequency': freq,
                    'expected': self.reference_frequencies[aa],
                    'severity': 'low'
                })

        print(anomalies)
        
        return {
            'feature_vector': target_features,
            'anomalies': anomalies
        }
    
    def detect_structure_anomalies(self, pdb_structure):
        """Detect anomalies in protein structures"""
        # Calculate features for the structure
        structure_features = self.calculate_structure_features(pdb_structure)
        
        anomalies = []
        
        # Check for unusual atom distances
        if 'ca_distance_mean' in structure_features:
            if structure_features['ca_distance_max'] > 50:  # Unusually large distance
                anomalies.append({
                    'type': 'unusual_extended_structure',
                    'max_distance': structure_features['ca_distance_max'],
                    'severity': 'medium'
                })
                
            if structure_features['ca_distance_std'] > 15:  # High variability in distances
                anomalies.append({
                    'type': 'unusual_structure_variability',
                    'distance_std': structure_features['ca_distance_std'],
                    'severity': 'medium'
                })
        
        # Check for unusually compact structures
        if 'ca_distance_mean' in structure_features and structure_features['ca_distance_mean'] < 10:
            anomalies.append({
                'type': 'unusual_compact_structure',
                'mean_distance': structure_features['ca_distance_mean'],
                'severity': 'medium'
            })
        
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
            clf = IsolationForest(contamination=0.1, random_state=42)
            clf.fit(X)
            
            # Transform target features
            target_vector = scaler.transform(np.array([list(target_features[col] for col in num_cols)]))
            
            # Get anomaly score
            anomaly_score = clf.score_samples(target_vector)[0]
            is_anomaly = clf.predict(target_vector)[0] == -1
            
            return {
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'threshold': clf.threshold_,
                'severity': 'high' if is_anomaly else 'none'
            }
        else:
            # Not enough data for ML, use simple PCA
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
    """Fetch protein data from UniProt"""
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
    """Fetch reference proteins based on a keyword"""
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