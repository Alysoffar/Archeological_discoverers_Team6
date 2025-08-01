import torch
from torch import nn
from torch.utils.data import Dataset

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer, OrdinalEncoder, StandardScaler
from sklearn.cluster import KMeans

import pandas as pd

class ArcheologicalDataset(Dataset):
    def __init__(self):
        self.df = pd.read_csv("dataset.csv")
        self.encode_timeperiod()
        self.encode_labels()

        numerical_feautures = [
            #"LocationCluster",
            "Human Activity Index",
            "Climate Change Impact",
            "Sonar Radar Detection",
            "Looting Risk (%)",
            "Period_Encoded"
        ]

        scaler = StandardScaler()
        scaled_numerical = torch.tensor(
            scaler.fit_transform(self.df[numerical_feautures].values), dtype=torch.float32
        )

        unscaled_features = [
            #"adobe", "arenisca", "bronce", "caliza", 
            #"granito", "ladrillo", "madera", "oro", "yeso", 
            "Script_encoded", #"Materials_encoded"
        ]
        unscaled_features = torch.tensor(self.df[unscaled_features].values, dtype=torch.float32)

        self.features = torch.cat([scaled_numerical, unscaled_features], dim=1)
        self.target = torch.tensor(self.df["AI Prediction Score"].values, dtype=torch.float32)

    @property
    def features_length(self):
        return self.features.shape[1]

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.target[idx]


    def encode_timeperiod(self):
        ordinal_order = [
            "Antiguo Reino",
            "Primer Período Intermedio",
            "Imperio Medio",
            "Segundo Período Intermedio",
            "Imperio Nuevo",
            "Tercer Período Intermedio",
            "Periodo Tardío",
            "Periodo Ptolemaico",
            "Periodo Romano"
        ]
        encoder = OrdinalEncoder(categories=[ordinal_order])
        self.df['Period_Encoded'] = encoder.fit_transform(self.df[['Time Period']])

    def encode_labels(self):
        self.df["Material Composition"] = self.df["Material Composition"].fillna("")
        material_lists = self.df["Material Composition"].str.lower().str.strip().str.split(r",\s*")
        
        material_ranking = {
            'yeso': 0.7, 
            'madera': 0.6,
            'ladrillo': 0.5,
            'oro': 1, 
            'bronce': 0.9,
            'granito': 0.85,
            'arenisca': 0.4,
            'adobe': 0.2, 
            'caliza': 0.8
        }
        #def apply_ranking(mat_list):
        #    return sum([material_ranking[mat] for mat in mat_list])
        #self.df["Materials_encoded"] = material_lists.apply(apply_ranking)

        mlb = MultiLabelBinarizer()
        multi_hot = mlb.fit_transform(material_lists)
        multi_hot_df = pd.DataFrame(multi_hot, columns=mlb.classes_, index=self.df.index)

        self.df = pd.concat([self.df, multi_hot_df], axis=1)
        #le = LabelEncoder()
        #self.df['Script_encoded'] = le.fit_transform(self.df['Script Detected'])
        scripts = {
            'Demótico': 0.9,
            'Cuneiforme':0.6,
            'Hierático': 0.85,
            'Griego': 0.7,
            'Copto': 0.8,
            'Jeroglífico': 1.0,
        }
        self.df['Script_encoded'] = self.df["Script Detected"].map(scripts)

        # Clustering the points
        coords = self.df[['Longitude', 'Latitude']].dropna()
        kmeans = KMeans(n_clusters=4, random_state=42)
        self.df['LocationCluster'] = kmeans.fit_predict(coords)

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_layer):
        super().__init__()
        
        self.fc1 = nn.Linear(input_layer, 100)
        self.dropout_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(100, 80)
        self.dropout_fc2 = nn.Dropout(p=0.5)
        self.output = nn.Linear(80, 1)

    def forward(self, input):
        act = self.fc1(input)
        act = torch.relu(act)
        act = self.dropout_fc1(act)
        act = self.fc2(act)
        act = torch.relu(act)
        act = self.dropout_fc2(act)
        return self.output(act)
        
#### train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from sklearn.metrics import r2_score

from model import NeuralNetworkModel, ArcheologicalDataset

EPOCH_NUM = 200
BATCH_SIZE = 5
DEVICE = "cuda"

def train():
    dataset = ArcheologicalDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    criterion = nn.HuberLoss()

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = NeuralNetworkModel(dataset.features_length).to(device="cuda")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("---------------- Starting Training --------------------")
    for epoch in range(EPOCH_NUM):
        model.train()
        all_preds = []
        all_targets = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch.reshape((BATCH_SIZE, 1)))
            loss.backward()
            optimizer.step()
            all_preds.append(y_pred.clone().detach().cpu())
            all_targets.append(y_batch.clone().detach().cpu())

        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()

        r2 = r2_score(targets, preds)

        print(f"Epoch {epoch+1}/{EPOCH_NUM} — R² Score: {r2:.4f}")

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(DEVICE)
            y_pred = model(x_batch).cpu()
            all_preds.append(y_pred)
            all_targets.append(y_batch)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    r2 = r2_score(targets, preds)
    print(f"Evaluation — R² Score: {r2:.4f}")


if __name__ == "__main__":
    train()
