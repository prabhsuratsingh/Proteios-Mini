import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.decomposition import PCA

from models.DrugDiscoveryVAE.VAE import MolecularVAE


def load_dd_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = MolecularVAE()
    vae.load_state_dict(torch.load("src\models\DrugDiscoveryVAE\molecular_vae_state_dict_2.pth", map_location=device))
    vae.to(device)
    vae.eval()

    return vae

def smiles_to_fp(smiles):
    generator = GetMorganGenerator(radius=2, fpSize=2048)
    mol = Chem.MolFromSmiles(smiles)
    return generator.GetFingerprint(mol) if mol else None

def prep_data():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_dd_model()
    df = pd.read_csv("src\models\DrugDiscoveryVAE\molecule_data.csv")
    df['fingerprints'] = df['smiles'].apply(smiles_to_fp)
    df = df.dropna()

    X = torch.tensor([list(fp) for fp in df['fingerprints']], dtype=torch.float32).to(device)
    y = torch.tensor(df['pIC50'].values, dtype=torch.float32).to(device)

    with torch.no_grad():
        mu, logvar = vae.encode(X)
        z = vae.reparameterize(mu, logvar)
        z_np = z.cpu().numpy()

    return (df, X, y, z, z_np)

def impl_pca():
    _, _, _, _, z_np = prep_data()
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_np)

    return z_pca

def create_fp(index):
    vae = load_dd_model()
    _, _, _, z, _ = prep_data()
    z_seed = z[index]
    noise = torch.randn_like(z_seed) * 0.2
    z_new = z_seed + noise
    new_fp = vae.decode(z_new).cpu().detach().numpy()

    return new_fp

