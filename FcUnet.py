import numpy as np
import scipy.io as sio
import mat73
import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import re


class SparseLinear(nn.Module): # https://stackoverflow.com/questions/63893602/neural-network-layer-without-all-connections
  def __init__(self, in_features, out_features, sparse_indices):
    super(SparseLinear, self).__init__()
    self.weight = nn.Parameter(data=torch.sparse.FloatTensor(sparse_indices, torch.randn(sparse_indices.shape[1]), [in_features, out_features]), requires_grad=True)
    self.bias = nn.Parameter(data=torch.randn(out_features), requires_grad=True)

  def forward(self, x):
    return torch.sparse.admm(self.bias, self.weight, x, 1., 1.)



class FcUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Fully connected layer to perform initial reconstruction
        self.fc1 = nn.Linear(1344, 80*64*32)

        # Encoder
        # input: 48*48*56*1
        self.e11 = nn.Conv3d(1, 16, kernel_size=3, padding='same')
        self.e12 = nn.Conv3d(16, 16, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        # input: 24*24*28*16
        self.e21 = nn.Conv3d(16, 32, kernel_size=3, padding='same')
        self.e22 = nn.Conv3d(32, 32, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        # input: 12*12*14*32
        self.e31 = nn.Conv3d(32, 64, kernel_size=3, padding='same')
        self.e32 = nn.Conv3d(64, 64, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        # input: 6*6*7*64
        self.b1 = nn.Conv3d(64, 128, kernel_size=3, padding='same')
        self.b2 = nn.Conv3d(128, 128, kernel_size=3, padding='same')

        # Decoder
        self.upconv1 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.d11 = nn.Conv3d(128, 64, kernel_size=3, padding='same')
        self.d12 = nn.Conv3d(64, 64, kernel_size=3, padding='same')

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.d21 = nn.Conv3d(64, 32, kernel_size=3, padding='same')
        self.d22 = nn.Conv3d(32, 32, kernel_size=3, padding='same')

        self.upconv3 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.d31 = nn.Conv3d(32, 16, kernel_size=3, padding='same')
        self.d32 = nn.Conv3d(16, 16, kernel_size=3, padding='same')

        # Output layer
        self.outconv = nn.Conv3d(16, 1, kernel_size=1)

    def forward(self, x):
        f1 = relu(self.fc1(x))
        f1_reshaped = f1.view(-1, 1, 32, 80, 64)
        # Encoder
        xe11 = relu(self.e11(f1_reshaped))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xb1 = relu(self.b1(xp3))
        xb2 = relu(self.b2(xb1))

        # Decoder
        xu1 = self.upconv1(xb2)
        xu11 = torch.cat([xe32, xu1], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xe22, xu2], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xe12, xu3], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        # Output layer
        out = self.outconv(xd32) + f1_reshaped
        return out


class mydata(Dataset):
    def __init__(self, X, Y, device):
        self.X = X
        self.Y = Y
        self.device = device

    def __len__(self):
        return self.Y.shape[-1]

    def __getitem__(self, idx):
        return torch.unsqueeze(self.X[:, idx], 0).to(self.device), torch.unsqueeze(self.Y[:, :, :, idx], 0).to(
            self.device)


def train_loop(dataloader, dataloader_test, model, mask, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        # loss = loss_fn(pred, y)
        loss = torch.zeros(1)
        if torch.cuda.is_available():
            loss = loss.to("cuda")
        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                tmp1 = torch.flatten(pred[i, j] * mask)
                tmp2 = torch.flatten(y[i, j] * mask)
                loss = loss.add(torch.sum((tmp1 - tmp2) ** 2))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            if np.isnan(loss):
                break
            print(f"loss: {loss:>7f}  [{current:>4d}/{size:>4d}]")

    model.eval()
    test_loss = torch.zeros(1)
    if torch.cuda.is_available():
        test_loss = test_loss.to("cuda")
    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader_test):
            X=X.to(device)
            y = y.to(device)
            pred = model(X)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    tmp1 = torch.flatten(pred[i, j] * mask)
                    tmp2 = torch.flatten(y[i, j] * mask)
                    test_loss = test_loss.add(torch.sum((tmp1 - tmp2) ** 2))

    test_loss = test_loss.item()
    print(f"test loss: {test_loss:>7f}")

    return loss.item(), test_loss


if __name__ == '__main__':
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    # device = ("cpu")
    print(f"Using {device} device")

    # data = sio.loadmat('../SimData/3D/images.mat')
    data_string = r'Datasets/FCModel1/Data_CCW1Mesh_spec4_2_nonan.mat'
    print(data_string)
    data = mat73.loadmat(data_string)
    data2 = mat73.loadmat(r'Datasets/FirstModel/images_CCW1Mesh_spec4_2_nonan.mat')
    print(data['all_dOD_noisy'].shape)
    print(data['all_dOD_noisy'][:1344, :1600].shape)
    print(data['all_dOD_noisy'][1344:, :1600].shape)


    training_X = torch.tensor(data['all_dOD_noisy'][:1344, :1600]/data['all_dOD_noisy'][1344:, :1600], dtype=torch.float32)
    training_Y = torch.tensor(data2['clean_images'][:, :, :, :1600], dtype=torch.float32)
    validation_X = torch.tensor(data['all_dOD_noisy'][:1344, 1600:1900]/data['all_dOD_noisy'][1344:, 1600:1900], dtype=torch.float32)
    validation_Y = torch.tensor(data2['clean_images'][:, :, :, 1600:1900], dtype=torch.float32)
    # inmesh = np.int16(data['inmesh'].squeeze())
    mask = torch.tensor(data2['mask'], dtype=torch.float32).to(device)

    model = FcUNet().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_dataloader = DataLoader(mydata(training_X, training_Y, device), batch_size=16)
    validate_dataloader = DataLoader(mydata(validation_X, validation_Y, device), batch_size=1)

    if device != 'cpu':
        torch.cuda.memory._record_memory_history(
            max_entries=100000
        )

    all_loss = []
    all_testloss = []
    mintest = np.inf
    for epoch in range(200):
        print(f'Epoch {epoch:>2d}')
        trainloss, testloss = train_loop(train_dataloader, validate_dataloader, model, mask, optimizer)
        all_loss.append(trainloss)
        all_testloss.append(testloss)
        if testloss < mintest:
            mintest = testloss
            patience = 10  # Reset patience counter
        # if epoch>5:
        #     if np.all(np.array(all_testloss[-5:])>mintest):
        #         print('Test loss exceeds minimum for 5 consecutive epochs. Terminating.')
        #         break
        else:
            patience -= 1
            if patience == 0:
                break

    if device != 'cpu':
        try:
            torch.cuda.memory._dump_snapshot(f"Datasets/FCModel1/memory_usage.pickle")
        except Exception as e:
            print(f"Failed to capture memory snapshot {e}")

            # Stop recording memory snapshot history.
        torch.cuda.memory._record_memory_history(enabled=None)


    model = model.to('cpu')
    model.eval()

    path_root_string = re.search('.*(?=\/)', data_string).group() + r'/'
    model_path = path_root_string + r'3D_UNet_trained3'
    torch.save(model, model_path)
    sio.savemat(path_root_string + 'loss_3D_UNet3.mat', {'training_loss': all_loss, 'testing_loss': all_testloss})

    # %%
    # Now process the test set
    model = torch.load(model_path)
    test_X = torch.tensor(data['noisy_images'][:1344, 1900:]/data['noisy_images'][1344:, 1900:], dtype=torch.float32)
    test_Y = np.zeros(test_X.shape)
    for i in range(test_X.shape[-1]):
        tmp = test_X[:, :, :, i]
        test_Y[:, :, :, i] = model(tmp.unsqueeze(0).unsqueeze(0)).squeeze().detach().numpy()

    sio.savemat(path_root_string + r'test_processed.mat', {'recon2': test_Y})