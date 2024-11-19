import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data.dataloader
import matplotlib.pylab as plt  

class RNN(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, 25)

    def forward(self, x):
        out, _ = self.rnn(x)
        result = out[:, 5:, :]
        result = self.relu(self.fc1(result))
        result = self.relu(self.fc2(result))
        final = self.fc3(result)
        return final

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(CustomDataset, self).__init__()
        self.x = x_tensor
        self.y = y_tensor

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        return x, y

# 데이터 로드 및 전처리
Data = pd.read_csv("C:/three.csv")
Data = Data.iloc[1:]
Data = Data.iloc[:, 1:]
Data = Data.values
data = Data.astype(np.float32)

X = np.array([data[i : i + 20] for i in range(154)])
Y = np.array([data[i + 10 : i + 25] for i in range(154)])
print(Y.shape)
"""X_max = np.max(X)
X_min = np.min(X)
X = (X - X_min) / (X_max - X_min)
print(X)"""
test = []
for i in Y:
    for j in range(15):
        for q in range(25):
            if(i[j][q] >= 100.0):
                test.append(1)
            else:
                test.append(0)
test = np.array(test)
Y = test.reshape(154, 15, 25)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)


X_train = torch.FloatTensor(X_train)
Y_train = torch.FloatTensor(Y_train)

dataset = CustomDataset(X_train, Y_train)

X_test = torch.FloatTensor(X_test)
Y_test = torch.FloatTensor(Y_test)

test_dataset = CustomDataset(X_test, Y_test)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

'''여기까지가 data 전처리'''

# 하이퍼 파라미터
layers = 3
in_size = 25
hid_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN(input_size=in_size, num_layers=layers,hidden_size=hid_size).to(device)

lr_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
loss_function = nn.BCEWithLogitsLoss()

num_epochs = 100

epoch_loss = []

for epoch in range(num_epochs):
    model.train()
    tot_loss = 0
    for x, y in train_loader:
        output = model(x)
        # output = torch.sigmoid(output)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tot_loss += loss.item()
    epoch_loss.append(tot_loss)
    if (epoch % 10 == 0):
        print(f"현재 epoch{epoch}의 loss값은: {tot_loss}")

torch.save(model.state_dict(), "test3.pth")
print("저장을 완료하였습니다.")
plt.plot(range(num_epochs), epoch_loss)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()      


model.load_state_dict(torch.load("test3.pth", weights_only=True))
model.eval()
with torch.no_grad():
    tot_loss = 0
    for x, y in test_loader:
        output = model(x)
        output = torch.sigmoid(output)
        output = torch.gt(output, 0.5).float()
        print(torch.eq(output, y))
        acc = torch.eq(output, y).float().sum() / torch.numel(output)

        # loss = loss_function(output, y)
        # tot_loss += loss.item()
        print(f"acc값은: {acc.item()}")
