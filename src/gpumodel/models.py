import math

import torch
from torch import nn

IMAGE_SHAPE = (3, 64, 64)
NUM_CLASSES = 10


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(math.prod(IMAGE_SHAPE), 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=3):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * ((c // self.heads) ** -0.5)
        x = x + self.proj((attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c))
        return x + self.ff(self.norm2(x))


class TinyViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, 192, kernel_size=8, stride=8)
        self.pos = nn.Parameter(torch.zeros(1, 64, 192))
        self.blocks = nn.Sequential(AttentionBlock(), AttentionBlock())
        self.norm = nn.LayerNorm(192)
        self.head = nn.Linear(192, NUM_CLASSES)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2) + self.pos
        return self.head(self.norm(self.blocks(x)).mean(dim=1))


MODELS = {"mlp": MLP, "cnn": CNN, "vit": TinyViT}


def make_model(name):
    return MODELS[name]()
