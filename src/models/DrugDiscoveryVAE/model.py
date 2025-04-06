import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator 
from rdkit.Chem import Descriptors

from VAE import MolecularVAE


# for .csv files
df = pd.read_csv("molecule_data.csv")

generator = GetMorganGenerator(radius=2, fpSize=2048)

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return generator.GetFingerprint(mol)
    return None

df['fingerprints'] = df['smiles'].apply(smiles_to_fp)
df = df.dropna()

def loss_function(recon_x, x, mu, logvar):
    bce_loss = nn.BCELoss()(recon_x, x)
    #KL-loss
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return bce_loss + kl_div


vae = MolecularVAE()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae.to(device)

#convert data to tensors for pytorch
X = torch.tensor([list(fp) for fp in df['fingerprints']], dtype=torch.float32).to(device)

losses = []

#train
num_epochs = 50
for epoch in range(num_epochs):
    vae.train()
    optimizer.zero_grad()
    
    recon_x, mu, logvar = vae(X)
    loss = loss_function(recon_x, X, mu, logvar)
    losses.append(loss.item())
    
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

vae.eval()
#imp - picks sample from latent space
z = torch.randn((5, 128)).to(device)
generated_fp = vae.decode(z).cpu().detach().numpy().flatten()



plt.plot(range(len(losses)), losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("VAE Training Loss Curve")
plt.legend()
plt.show()

#rd-kit convert fp to molecule
def fingerprint_to_smiles(fp_array):
    bit_vector = DataStructs.ExplicitBitVect(len(fp_array))
    for i, bit in enumerate(fp_array):
        if bit > 0.5: 
            bit_vector.SetBit(i)

    df['similarity'] = df['fingerprints'].apply(lambda fp: DataStructs.TanimotoSimilarity(fp, bit_vector))
    best_match = df.loc[df['similarity'].idxmax()]
    
    return best_match['smiles']  

print("Generated Molecule:", fingerprint_to_smiles(generated_fp))
gen_mol = fingerprint_to_smiles(generated_fp)


torch.save(vae.state_dict(), "molecular_vae.pth")
torch.save(vae, "molecular_vae_model.pth")
print("Model saved successfully!")


def analyze_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    
    if mol:
        mw = Descriptors.MolWt(mol)  
        logp = Descriptors.MolLogP(mol)  
        num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)  
        num_rot_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        
        print(f"Generated Molecule: {smiles}")
        print(f"  ✅ Valid Molecule")
        print(f"  🔹 Molecular Weight: {mw:.2f}")
        print(f"  🔹 LogP: {logp:.2f}")
        print(f"  🔹 Rings: {num_rings}, Rotatable Bonds: {num_rot_bonds}")

        # Draw.MolToImage(mol).show()
    else:
        print(f"❌ Invalid Molecule: {smiles}")

analyze_molecule(gen_mol)


