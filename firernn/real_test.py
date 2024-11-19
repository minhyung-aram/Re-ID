import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib.pylab as plt
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# 하이퍼 파라미터
layers = 3
hid_size = 90
future_step = 5
encode_length = 0
seq_length = 30
stride = 1
in_size = 56

class RNN(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, input_size)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.3)
        # self.bn1 = nn.BatchNorm1d(seq_length)
        # self.bn2 = nn.BatchNorm1d(seq_length)
        # self.fc2 = nn.Linear(hidden_size // 4, hidden_size // 8)
        # self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        # self.fc4 = nn.Linear(hidden_size // 8, 25)

    def forward(self, x):
        out, (hn, cn) = self.rnn(x)
        result = out[:, encode_length:, :]
       # result = self.relu(self.fc1(result))
       # result = self.relu(self.fc3(result))
        final = self.fc1(result)
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

def preprocess(Data):
    Data = Data.iloc[1:]
    Data = Data.iloc[:, 1:]
    Data = Data.values
    data = Data.astype(np.float32)

    X = np.array([data[i : i + seq_length] for i in range(0, data.shape[0] - seq_length - 5, stride) if i + seq_length + future_step <= data.shape[0]])
    Y = np.array([data[i + encode_length + future_step : i + seq_length + future_step] for i in range(0, data.shape[0] - seq_length, stride) if i + seq_length + future_step <= data.shape[0]])
    assert (X.shape[1] - encode_length == Y.shape[1]), "사이즈 확인하세요!"
    return X, Y

Data = pd.read_csv("university_no_door_devc.csv")
Data = Data.iloc[1:]
Data = Data.iloc[:, 1:]
Data = Data.values
data = Data.astype(np.float32)
data = data[:800, :]
data = scaler.fit_transform(data)

X = np.array([data[i : i + seq_length] for i in range(0, data.shape[0] - seq_length - 5, stride) if i + seq_length + future_step <= data.shape[0]])
Y = np.array([data[i + encode_length + future_step : i + seq_length + future_step] for i in range(0, data.shape[0] - seq_length, stride) if i + seq_length + future_step <= data.shape[0]])
Y = Y[:-1, ...]
"""X_max = np.max(X)
X_min = np.min(X)
X = (X - X_min) / (X_max - X_min)
print(X)"""

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0, random_state=42)


X_train = torch.FloatTensor(X)
Y_train = torch.FloatTensor(Y)

dataset = CustomDataset(X_train, Y_train)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN(input_size=in_size, num_layers=layers, hidden_size=hid_size)
model = model.to(device)
lr_rate = 0.0001 # 0.001도 한번 해봐
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
loss_function = nn.HuberLoss(delta=1)

num_epochs = 200 # 너무 적어
epoch_loss = []

for epoch in range(num_epochs):
    model.train()
    tot_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        #output = torch.sigmoid(output)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        tot_loss += loss.item()
    epoch_loss.append(tot_loss)
    if (epoch % 10 == 0):
        print(f"현재 epoch{epoch}의 loss값은: {tot_loss}")

torch.save(model.state_dict(), "BCE1.pth")
# print("저장을 완료하였습니다.")
# plt.plot(range(num_epochs), epoch_loss)
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()

Data = pd.read_csv("university_51_55sec.csv")
Data = Data.iloc[1:]
Data = Data.iloc[:, 1:]
Data = Data.values
data = Data.astype(np.float32)
data = scaler.fit_transform(data)

valid_X = np.array([data[i : i + seq_length] for i in range(0, data.shape[0] - seq_length - 5, stride) if i + seq_length + future_step <= data.shape[0]])
valid_Y = np.array([data[i + encode_length + future_step : i + seq_length + future_step] for i in range(0, data.shape[0] - seq_length, stride) if i + seq_length + future_step <= data.shape[0]])
valid_Y = valid_Y[:-1, ...]

valid_x = torch.FloatTensor(valid_X)
valid_y = torch.FloatTensor(valid_Y)

valid_data = CustomDataset(valid_x, valid_y)

valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=1, shuffle=False)

model.load_state_dict(torch.load("BCE1.pth", weights_only=True))
model.eval()
num = 0
with torch.no_grad():
    tot_loss = []
    for x, y in valid_loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        print(f"모델의 예측: {output} \n \n 실제 레이블: {y}")
        loss = loss_function(output, y)
        # loss = loss_function(output, y)
        tot_loss.append(loss.item())
        num += 1

plt.plot(range(num), tot_loss)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
plt.imshow()