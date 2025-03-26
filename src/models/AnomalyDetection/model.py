import numpy as np
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
        return self.scaler.transform(X)
    
    def fit(self, X, validation_split=0.1, epochs=100, batch_size=32):
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_val = train_test_split(X_scaled, test_size=validation_split)
        
        if self.autoencoder is None:
            self.build_model(X_train.shape[1])
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = self.autoencoder.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        reconstructions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        self.threshold = np.percentile(mse, 95)
        
        return history
    
    def detect_anomalies(self, X, return_scores=False):
        X_scaled = self.preprocess_data(X)
        reconstructions = self.autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - reconstructions), axis=1)
        anomalies = mse > self.threshold
        
        if return_scores:
            return anomalies, mse
        return anomalies
    
    def extract_features(self, X):
        X_scaled = self.preprocess_data(X)
        return self.encoder.predict(X_scaled)
    
    def visualize_anomalies(self, X, labels=None):
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

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(mse)), mse, c=labels, cmap='coolwarm', alpha=0.7)
        plt.axhline(self.threshold, color='r', linestyle='--', label=f'Threshold: {self.threshold:.5f}')
        plt.xlabel('Protein Index')
        plt.ylabel('Reconstruction Error')
        plt.title('Anomaly Scores for Proteins')
        plt.legend()
        plt.show()
        
        # if labels is not None:
        #     plt.figure(figsize=(10, 6))
        #     plt.scatter(range(len(mse)), mse, c=labels, cmap='coolwarm', alpha=0.7)
        #     plt.axhline(self.threshold, color='r', linestyle='--', label=f'Threshold: {self.threshold:.5f}')
        #     plt.xlabel('Protein Index')
        #     plt.ylabel('Reconstruction Error')
        #     plt.title('Anomaly Scores for Proteins')
        #     plt.legend()
        #     plt.show()
