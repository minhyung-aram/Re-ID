import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import time 
import matplotlib.pylab as plt  

# 하이퍼 파라미터
layers = 3
hid_size = 64
future_step = 5
encode_length = 0
seq_length = 40
stride = 1
in_size = 25
num_classes = 3
out_size = in_size * num_classes

class RNN(nn.Module):
    def __init__(self, num_layers, hidden_size, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, dropout=0.1, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, out_size)
        self.relu = nn.ReLU()
        self.drop_out = nn.Dropout(0.4)
        # self.bn1 = nn.BatchNorm1d(seq_length)
        # self.bn2 = nn.BatchNorm1d(seq_length)
        # self.fc2 = nn.Linear(hidden_size // 4, hidden_size // 8)
        # self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        # self.fc4 = nn.Linear(hidden_size // 8, 25)

    def forward(self, x):
        out, _ = self.rnn(x)
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

    X = np.array([data[i : i + seq_length] for i in range(0, data.shape[0] - seq_length - future_step, stride) if i + seq_length + future_step <= data.shape[0]])
    Y = np.array([data[i + encode_length + future_step : i + seq_length + future_step] for i in range(0, data.shape[0] - seq_length, stride) if i + seq_length + future_step <= data.shape[0]])
    assert (X.shape[1] - encode_length == Y.shape[1]), "사이즈 확인하세요!"
    print(X.shape, Y.shape)
    test = []
    for i in Y:
        for j in range(seq_length - encode_length):
            for q in range(25):
                if(i[j][q] >=150.0):
                    test.append(1)
                else:
                    test.append(0)
    test = np.array(test, dtype=np.float32)
    Y = test.reshape(Y.shape)
    return X, Y

Data1 = pd.read_csv("C:/kim.csv")
X_1, Y_1 = preprocess(Data1)

Data2 = pd.read_csv("C:/min.csv")
X_2, Y_2 = preprocess(Data2)

train_x = np.vstack((X_1, X_2))
train_y = np.vstack((Y_1, Y_2))

print("총 데이터 량: ", train_x.shape[0])
X_train = torch.FloatTensor(train_x)
Y_train = torch.FloatTensor(train_y)

dataset = CustomDataset(X_train, Y_train)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNN(input_size=in_size, num_layers=layers, hidden_size=hid_size).to(device)

lr_rate = 0.001 # 0.001도 한번 해봐
optimizer = torch.optim.Adam(model.parameters(), lr=lr_rate)
loss_function = nn.BCEWithLogitsLoss()

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

# torch.save(model.state_dict(), "test_fire_bce1.pth")
# print("저장을 완료하였습니다.")
# plt.plot(range(num_epochs), epoch_loss)
# plt.xlabel("epochs")
# plt.ylabel("loss")
# plt.show()      

Data3 = pd.read_csv("C:/hyung.csv")
X_3, Y_3 = preprocess(Data3)

valid_x = torch.FloatTensor(X_3)
valid_y = torch.FloatTensor(Y_3)
valid_dataset = CustomDataset(valid_x, valid_y)
valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False)
# model.load_state_dict(torch.load("test_fire_bce.pth", weights_only=True))
model.eval()
with torch.no_grad():
    tot_loss = 0
    for x, y in valid_data_loader:
        output = model(x)
        output = torch.sigmoid(output)
        print("모델 output: ", output)
        time.sleep(10)
        output = torch.gt(output, 0.5).float()
        # print(torch.eq(output, y))
        acc = torch.eq(output, y).float().sum() / torch.numel(output)
        print("acc: ", acc)
        time.sleep(3)
        # loss = loss_function(output, y)
        # tot_loss += loss.item()
        print("acc값: ", acc)
