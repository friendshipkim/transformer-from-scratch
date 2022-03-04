import torch
from model.layer.embedding import Embedding


def test_embedding():
    # set params
    d_model = 256
    vocab_size = 10000
    batch_size = 16
    max_seq_len = 10

    embedding = Embedding(d_model=d_model, vocab_size=vocab_size, max_seq_len=max_seq_len)

    # inputs
    x = torch.randint(vocab_size, (batch_size, max_seq_len))

    out = embedding(x)

    # TODO: what should I check?

    return out

if __name__ == '__main__':
    test_embedding()
