import torch
from torch import nn

from model.encoder import Encoder

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.input_emb =