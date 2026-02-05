import torch
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dat_path = Path(__file__).parent/'data' 

dat = pd.read_csv(dat_path / 'train.csv')
dat = pd.get_dummies(dat)

class dubleconv(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.block(x)


class encoder(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()

        self.conv = dubleconv(input_channels,output_channels)
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self,x):
        conv = self.conv(x)
        pool = self.pool(conv)
        return conv, pool


class extractor(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.enc1 = encoder(1,64)
        self.enc2 = encoder(64,128)
        self.enc3 = encoder(128,256)
        self.enc4 = encoder(256,512)
        self.enc5 = dubleconv(512,1024)
        
        self.lin1 = nn.Linear(1024,512)
        self.lin2 = nn.Linear(512,256)
        self.lin3 = nn.Linear(256,128)
        self.lin4 = nn.Linear(128,64)
        self.lin5 = nn.Linear(64,1)

    def forward(self,x):

        s1,x = self.enc1(x)
        s2,x = self.enc2(x)
        s3,x = self.enc3(x)
        s4,x = self.enc4(x)

        x = self.enc5(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        x = self.lin5(x)
        return x

#model = extractor()


X = dat.drop(columns=['exam_score']).values
y = dat['exam_score'].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
model = extractor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    mse = criterion(predictions, y_test_tensor)
    print(f"Test MSE: {mse.item():.4f}")
