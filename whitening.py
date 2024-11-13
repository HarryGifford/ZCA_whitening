'''
Implements a whitening normalization layer for PyTorch.

Whitening is a transformation that removes the correlation between features.
It does this by
1. Centering the data by subtracting the mean of each feature.
2. Computing the covariance matrix of the centered data.
3. Applying an eigenvalue decomposition to the covariance matrix, C = PDP^T.
4. Computing W_pca = P * D^(-1/2) * P^T.
5. Multiplying the centered data by W_pca.
'''

import torch.linalg
import torch
import torch.nn as nn

from show_images import *

class WhiteningNormalization(nn.Module):
    '''
    Whitening normalization layer.
    '''
    def __init__(self, num_features: int, eps = 1e-3):
        '''
        Constructor.

        num_features: int
            The number of features in the input.
        eps: float
            A small value to prevent instability caused by
            very small eigenvalues.
        '''
        super(WhiteningNormalization, self).__init__()
        self.mean = torch.zeros(num_features)
        self.eigvecs = torch.eye(num_features)
        self.eigvals = torch.ones(num_features)
        self.w_zca = None
        self.eps = eps
        self.n_samples = 0
        torch.randn(1000, 10)
    
    def compute_zca(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Compute the ZCA matrix.
        '''
        x_centered = x - self.mean
        cov = x_centered.T @ x_centered / x.size(0)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        w_zca = eigvecs @ torch.diag(1 / torch.sqrt(eigvals + self.eps)) @ eigvecs.T
        return w_zca, eigvals, eigvecs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass.
        '''
        # If we're in training mode, compute the mean and covariance matrix.
        # We compute a running average of the mean and covariance matrix.
        if self.training:
            factor = self.n_samples / (self.n_samples + x.size(0))
            self.n_samples += x.size(0)
            if factor > 0.999:
                # If we have enough samples, we can just use the stored mean and covariance matrix.
                x_centered = x - self.mean
                if self.w_zca is None:
                    w_zca = self.eigvecs @ torch.diag(1 / torch.sqrt(self.eigvals + self.eps)) @ self.eigvecs.T
                    self.w_zca = w_zca
                else:
                    w_zca = self.w_zca
            else:
                batch_mean = x.mean(dim=0)
                x_centered = x - batch_mean
                batch_cov = (x_centered).T @ (x_centered) / x.size(0)
                # Update the mean.
                self.mean = factor * self.mean + (1 - factor) * batch_mean
                # Update the covariance matrix.
                cov = factor * (self.eigvecs @ torch.diag(self.eigvals + self.eps) @ self.eigvecs.T) + (1 - factor) * batch_cov
                # Apply eigenvalue decomposition.
                eigvals, eigvecs = torch.linalg.eigh(cov)
                self.eigvals = eigvals
                self.eigvecs = eigvecs
                w_zca = eigvecs @ torch.diag(1 / torch.sqrt(eigvals + self.eps)) @ eigvecs.T
        else:
            if self.w_zca is None:
                if self.eigvals is None or self.eigvecs is None:
                    raise ValueError("No eigenvectors and eigenvalues have been computed.")
                w_zca = self.eigvecs @ torch.diag(1 / torch.sqrt(self.eigvals + self.eps)) @ self.eigvecs.T
                self.w_zca = w_zca
            else:
                w_zca = self.w_zca
        x_centered = x - self.mean
        return x_centered @ w_zca

if __name__ == "__main__":
    # Test the whitening normalization layer with the CIFAR-10 dataset.
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    # Load the CIFAR-10 dataset.
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Convert the images to grayscale.
        transforms.Grayscale(num_output_channels=1),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32*32*4, shuffle=True, num_workers=2)

    original_images, _ = next(iter(trainloader))
    original_images = original_images[:100]
    show_images(original_images, 32, 32, 1, "assets/original.png")

    # Create the whitening normalization layer.
    whitening = WhiteningNormalization(1 * 32 * 32)
    whitening.train()
    # Iterate through the CIFAR-10 dataset.
    for images, _ in iter(trainloader):
        images = images.view(images.size(0), -1)
        # Forward pass.
        whitened_images = whitening(images)
    whitening.eval()
    whitened_images = whitening(original_images.view(100, -1))
    show_images(whitened_images, 32, 32, 1, "assets/whitened.png")
