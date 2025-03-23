import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.models import Model, Sequential
from keras._tf_keras.keras.layers import Input, Dense, Dropout
from keras._tf_keras.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

class ProteinAnomalyDetector:
    def __init__(self, encoding_dim=32, hidden_dim=64):
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.autoencoder = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.threshold = None
    
    def build_model(self, input_dim):
        """Build the autoencoder model architecture"""
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(self.hidden_dim, activation='relu')(input_layer)
        encoder = Dropout(0.2)(encoder)
        encoder = Dense(self.encoding_dim, activation='relu')(encoder)
        
        # Decoder
        decoder = Dense(self.hidden_dim, activation='relu')(encoder)
        decoder = Dropout(0.2)(decoder)
        decoder = Dense(input_dim, activation='sigmoid')(decoder)
        
        # Autoencoder
        self.autoencoder = Model(input_layer, decoder)
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        # Encoder model for feature extraction
        self.encoder = Model(input_layer, encoder)
        
        return self.autoencoder
    
    def preprocess_data(self, X):
        """Preprocess protein feature data"""
        return self.scaler.transform(X)
    
    def fit(self, X, validation_split=0.1, epochs=100, batch_size=32):
        """Train the autoencoder on normal protein data"""
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val = train_test_split(X_scaled, test_size=validation_split)
        
        # Build model if not already built
        if self.autoencoder is None:
            self.build_model(X_train.shape[1])
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Set anomaly threshold based on reconstruction error
        reconstructions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        self.threshold = np.percentile(mse, 95)  # 95th percentile as threshold
        
        return history
    
    def detect_anomalies(self, X, return_scores=False):
        """Detect anomalies in protein data"""
        X_scaled = self.preprocess_data(X)
        reconstructions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        anomalies = mse > self.threshold
        
        if return_scores:
            return anomalies, mse
        return anomalies
    
    def extract_features(self, X):
        """Extract encoded features from protein data"""
        X_scaled = self.preprocess_data(X)
        return self.encoder.predict(X_scaled)
    
    def visualize_anomalies(self, X, labels=None):
        """Visualize anomaly scores"""
        X_scaled = self.preprocess_data(X)
        reconstructions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.hist(mse, bins=50, alpha=0.7)
        plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold: {self.threshold:.5f}')
        plt.xlabel('Reconstruction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Reconstruction Errors')
        plt.legend()
        plt.show()
        
        if labels is not None:
            plt.figure(figsize=(10, 6))
            plt.scatter(range(len(mse)), mse, c=labels, cmap='coolwarm', alpha=0.7)
            plt.axhline(self.threshold, color='r', linestyle='--', label=f'Threshold: {self.threshold:.5f}')
            plt.xlabel('Protein Index')
            plt.ylabel('Reconstruction Error')
            plt.title('Anomaly Scores for Proteins')
            plt.legend()
            plt.show()

# Example usage
if __name__ == "__main__":
    # Load your protein features data
    # This could be embeddings from ESM-2 or manually engineered features
    # X = load_protein_features("protein_features.csv")
    
    # For demonstration, let's create synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100
    
    # Generate normal protein data
    X_normal = np.random.normal(0, 1, size=(n_samples, n_features))
    
    # Generate anomalous protein data
    X_anomalous = np.random.normal(2, 1, size=(int(n_samples*0.1), n_features))
    
    # Combine data with labels for evaluation
    X_combined = np.vstack([X_normal, X_anomalous])
    y_true = np.concatenate([np.zeros(n_samples), np.ones(int(n_samples*0.1))])
    
    # Create and train the model
    detector = ProteinAnomalyDetector(encoding_dim=32, hidden_dim=64)
    history = detector.fit(X_normal, epochs=50)
    
    # Detect anomalies
    anomalies, scores = detector.detect_anomalies(X_combined, return_scores=True)
    
    # Visualize results
    detector.visualize_anomalies(X_combined, labels=y_true)
    
    # Calculate performance metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    precision = precision_score(y_true, anomalies)
    recall = recall_score(y_true, anomalies)
    f1 = f1_score(y_true, anomalies)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")