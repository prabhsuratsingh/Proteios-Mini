import numpy as np
import pandas as pd
from keras._tf_keras.keras.saving import load_model
import pickle

from models.AnomalyDetection.model import ProteinAnomalyDetector

def prepare_protein_data(sequences, feature_type="composition"):
    features = []
    
    if feature_type == "composition":
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        for seq in sequences:
            composition = [seq.count(aa)/len(seq) for aa in amino_acids]
            features.append(composition)
    
    elif feature_type == "embeddings":
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
            model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
            
            for seq in sequences:
                seq = seq[:1022] if len(seq) > 1022 else seq
                
                inputs = tokenizer(seq, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                
                embedding = outputs.last_hidden_state[:, 0, :].numpy().flatten()
                features.append(embedding)
        except ImportError:
            print("Error: Required libraries not installed for embeddings")
            return None
    
    return np.array(features)

def load_model_project():
    autoencoder = load_model("src/models/AnomalyDetection/autoencoder.keras")
    encoder = load_model("src/models/AnomalyDetection/encoder.keras")

    with open('src/models/AnomalyDetection/model_metadata.pkl', 'rb') as f:
        model_data = pickle.load(f)

    detector = ProteinAnomalyDetector()

    detector.autoencoder = autoencoder
    detector.encoder = encoder
    detector.scaler = model_data['scaler']
    detector.threshold = model_data['threshold']
    
    return detector
