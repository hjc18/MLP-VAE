import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch import Tensor
from typing import List


class GaussianVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, x_var=1):
        super(GaussianVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.z_mu = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
        )
        self.x_mu = nn.Linear(hidden_dim, input_dim)
        # self.x_logvar = nn.Linear(hidden_dim, input_dim)
        self.x_var = x_var

    def encode(self, input: Tensor) -> List[Tensor]:
        input = self.encoder(input)
        mu = self.z_mu(input)
        logvar = self.z_logvar(input)
        return [mu, logvar]
    
    def decode(self, input: Tensor):
        input = self.decoder(input)
        mu = self.x_mu(input)
        # logvar = self.x_logvar(input)
        return mu
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = x.flatten(start_dim=1)
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_z = self.decode(z)
        return [x, z_mu, z_logvar, x_z]
    
    def loss(self, x: Tensor, z_mu: Tensor, z_logvar: Tensor, x_z: Tensor) -> Tensor:
        kldiv = -0.5 * torch.mean(torch.sum(1 + z_logvar - z_mu * z_mu - z_logvar.exp(), dim=1))
        recon = F.mse_loss(x, x_z) / (2*self.x_var) * self.input_dim
        return recon + kldiv
    
    def sample(self, n):
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)


class BernoulliVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(BernoulliVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.z_mu = nn.Linear(hidden_dim, latent_dim)
        self.z_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        input = self.encoder(input)
        mu = self.z_mu(input)
        logvar = self.z_logvar(input)
        return [mu, logvar]
    
    def decode(self, input: Tensor):
        input = self.decoder(input)
        return input
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = (x.flatten(start_dim=1) > 0).float()
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_z = self.decode(z)
        return [x, z_mu, z_logvar, x_z]
    
    def loss(self, x: Tensor, z_mu: Tensor, z_logvar: Tensor, x_z: Tensor) -> Tensor:
        kldiv = -0.5 * torch.mean(torch.sum(1 + z_logvar - z_mu * z_mu - z_logvar.exp(), dim=1))
        # recon = F.mse_loss(x, x_z) / (2*self.x_var) * self.input_dim
        recon = F.binary_cross_entropy(x_z, x) * self.input_dim
        return recon + kldiv
    
    def sample(self, n):
        z = torch.randn(n, self.latent_dim)
        return self.decode(z)



def draw(filepath, data: Tensor, n):
    fig = plt.figure(figsize=[8, 1], layout='compressed')
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(data[i].view(28, 28), cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    fig.savefig(filepath)
    plt.close()


def draw_contrast(filepath, x, z_mu, z_logvar, x_z, n):
    fig = plt.figure(figsize=[8, 3], layout='constrained')
    for i in range(n):
        plt.subplot(3, n, i+1)
        plt.imshow(x[i].view(28, 28), cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3, n, n+i+1)
        plt.imshow(x_z[i].view(28, 28), cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(3, n, 2*n+i+1)
        plt.imshow(torch.cat((z_mu[i], z_logvar[i].exp()), dim=0).view(8, 10), cmap='viridis', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    fig.savefig(filepath)
    plt.close()


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        x, z_mu, z_logvar, x_z = model(data)
        loss = model.loss(x, z_mu, z_logvar, x_z)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x, z_mu, z_logvar, x_z = model(data)
            test_loss += model.loss(x, z_mu, z_logvar, x_z).item() * len(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss        

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))




def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--gaussian-var', type=float, default=1,
                        help='the variance of Gaussian x|z')
    parser.add_argument('--bernoulli', action='store_true', default=False,
                        help='assuming x|z as Bernoulli')
    parser.add_argument('--sample-dir', type=str, default='sample')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available() # mps is not stable on Apple M1

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # model = Net().to(device)
    # model = GaussianVAE(28*28, 512, 40).to(device)
    if args.bernoulli:
        model = BernoulliVAE(28*28, 512, 40).to(device)
    else:
        model = GaussianVAE(28*28, 512, 40, args.gaussian_var).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    os.makedirs(args.sample_dir)
    for epoch in range(1, args.epochs + 1):
        print("current learning rate: {}".format(scheduler.get_last_lr()))
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        with torch.no_grad():
            draw("./{}/epoch{}.png".format(args.sample_dir, epoch), model.sample(8), 8)
            example = iter(test_loader)
            (data, _) = next(example)
            x, z_mu, z_logvar, x_z = model(data)
            draw_contrast("./{}/contrast_epoch{}.png".format(args.sample_dir, epoch), x, z_mu, z_logvar, x_z, 12)
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()