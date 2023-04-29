from datetime import datetime

import torch
import torch.nn.functional as F
from dataset import loadData
from sklearn.preprocessing import StandardScaler
from torch import nn
from utils import *

device = "cpu"

class PathPredictor(nn.Module):
    """
    Version: 2023.04.25.
    
    """
    def __init__(self, MLP_OUT=1024, CNN_OUT=512):
        super(PathPredictor, self).__init__()

        self.MLP_OUT = MLP_OUT
        self.CNN_OUT = CNN_OUT
        self.n_features = MLP_OUT + CNN_OUT

        setRandomSeed(42)
        self.MLP = nn.Sequential(
            nn.Linear(6, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, MLP_OUT),
            nn.LayerNorm(MLP_OUT),
            nn.ReLU(inplace=True),
        )

        self.CNN = nn.Sequential(
            nn.Conv2d(10, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 512, kernel_size=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, CNN_OUT, kernel_size=2),
            nn.LayerNorm([512, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        
        self.Attention = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.Sigmoid()
        )

        self.predictor = nn.Sequential(
            nn.Linear(self.n_features, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 2)
        )

        self.predictor2 = nn.Sequential(
            nn.Linear(self.n_features * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2)
        )

        self.init_weights()
        self.scaler = StandardScaler()

    def init_weights(self):
        for submodule in self.children():
            if isinstance(submodule, nn.Sequential):
                for layer in submodule:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.constant_(layer.bias, 0.)
                    if isinstance(layer, nn.Conv2d):
                        nn.init.kaiming_uniform_(layer.weight)
                        nn.init.constant_(layer.bias, 0.)

    def extractFeatures(self, Xb, Xe):
        Fb = self.MLP(Xb)
        Fe = self.CNN(Xe)

        Ff = torch.cat([Fb, Fe], axis=1)

        w = self.Attention(Ff)
        return Ff * w

    def forward(self, x, query_hour):
        x_cur = self.scaler.inverse_transform(x[:, 1])
        x_bef = self.scaler.inverse_transform(x[:, 0])

        Xb = x[:, :, :6].to(device)  # (n, 2, 6)
        Xe = x[:, :, 7:].reshape([-1, 2, 3, 3, 10]).permute(0, 1, 4, 2, 3).to(device)  # (n, 2, 10, 3, 3)

        F_cur = self.extractFeatures(Xb[:, 1], Xe[:, 1])
        F_bef = self.extractFeatures(Xb[:, 0], Xe[:, 0])

        interval = x_cur[:, 2] - x_bef[:, 2]
        for i, _ in enumerate(interval):
            if interval[i]==0:
                interval[i]=0.001
            if interval[i] < 0: interval[i]+=24
        interval = torch.Tensor(interval).to(device)

        df = (F_cur-F_bef) / interval.unsqueeze(1) * query_hour.unsqueeze(1).to(device)
        x_cur = torch.Tensor(x_cur).to(device)
        #return self.predictor(df) + x_cur[:, 4:6]
        f_next = torch.cat([F_cur, df], axis=1)
        return self.predictor2(f_next) + x_cur[:, 4:6]

    def set_optimizer(self, learning_rate, weight_decay):
        self.criterion = nn.MSELoss().to(device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def toTensor(self, arrays):
        if len(arrays)==1:
            return torch.tensor(arrays[0], dtype=torch.float)
        return [torch.tensor(arr, dtype=torch.float) for arr in arrays]

    def fit(self, x, y, training_epochs=100, batch_size=4096, learning_rate=0.0001, weight_decay= 1e-5, save_model=False):
        print("Start Training Model...")
        self.set_optimizer(learning_rate, weight_decay)

        self.scaler.fit(x[:, 0])
        self.scaler.fit(x[:, 1])
        x[:, 0] = self.scaler.transform(x[:, 0])
        x[:, 1] = self.scaler.transform(x[:, 1])

        Y = y[:, 4:7]
        X, Y = self.toTensor([x, Y])

        data_loader = torch.utils.data.DataLoader(dataset=torch.utils.data.TensorDataset(X, Y),
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  drop_last=True)
        self.train()
        for epoch in range(1, training_epochs+1):
            avg_cost = 0
            total_batch = len(data_loader)
            
            for x_batch, y_batch in data_loader:
                y_batch = y_batch.to(device)
                self.optimizer.zero_grad()
                hypothesis = self.forward(x_batch, y_batch[:, 2])
                cost = self.criterion(hypothesis, y_batch[:, :2])
                cost.backward()
                self.optimizer.step()
                
                avg_cost += cost
            avg_cost /= total_batch
            if epoch%10 == 0 or True:
                print("Epoch:" , epoch, "cost:", avg_cost.item())
        print("Training Complete!")
        if save_model:
            torch.save(model, "./checkpoints/checkpoint.pth", map_location=torch.device('cpu'))
        
    def predict(self, x, target_hour=None):
        x[:, 0] = self.scaler.transform(x[:, 0])
        x[:, 1] = self.scaler.transform(x[:, 1])
        X = self.toTensor([x])

        if target_hour is None:
            target_hour = torch.full((X.shape[0],), 6.)
        target_hour = torch.Tensor(target_hour)

        self.eval()
        with torch.no_grad():
            prediction = self.forward(X, target_hour)
        return prediction.to("cpu")
    
if __name__ == "__main__":
    x_train, y_train, x_test, y_test = loadData(path_data=True, augment=1)
    
    model = PathPredictor().to(device)
    model.fit(x_train, y_train, training_epochs=30, batch_size=4096, learning_rate=0.001, weight_decay= 1e-5, save_model=False)
    
    pred = model.predict(x_test, y_test[:, 6])
    from sklearn.metrics import mean_squared_error
    rmse = mean_squared_error(pred, y_test[:, 4:6], squared=False)
    diff = [distance(pred[i][0], pred[i][1], y_test[i][4], y_test[i][5]) for i in range(len(y_test))]

    print("Test Score:", rmse)
    print("Distance Score: %.3f (%.3f)"%(np.mean(diff), np.std(diff)))

    print(pred[0])
    print(y_test[0][4:6])
