import os.path

import config as cfg  # hyperparameters
from model.transformer import Transformer

import torchtext.datasets
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
import torch
from torch import nn, Tensor
from typing import Tuple
import numpy as np

import copy
import time
import math


def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat 1D Tensor."""
    data = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
        for item in raw_text_iter
    ]
    # vocab method maps tokens to indices.
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))  # actually filtering isn't necessary
    # TODO: why tuple - to use torch.cat, why there exist case t.numel() = 0 (blank line)
    # line.strip() is empty, don't read that line

def batchify(data: Tensor, batch_size: int) -> Tensor:
    """Divides the data into batch_size separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        batch_size: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // batch_size
    data = data[: seq_len * batch_size]  # truncate data
    data = data.view(batch_size, seq_len).t().contiguous()  # shape [seq_len, batch_size] # TODO: transpose??
    return data.to(cfg.device)

def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [max_seq_len, batch_size] and
        target has shape [max_seq_len * batch_size]
    """
    seq_len = min(cfg.max_seq_len, len(source) - 1 - i)
    # if shorter than max_seq_len, padding
    if seq_len < cfg.max_seq_len:
        print("sequence length is too short, padding:", seq_len)
        return source[i:i + seq_len].t(), source[i:i + seq_len].t()
        # padding = torch.ones(cfg.max_seq_len-seq_len, source.size(1)).to(cfg.device)
        # # print(padding.shape)
        # data = torch.cat((source[i:i + seq_len], padding), axis=0).long().t() # TODO: check: transposed
        # print(data.shape)
        # target = data.reshape(-1)
        # return data, target

    data = source[i:i + seq_len].t()  # shape: [max_seq_len, batch_size] TODO: why transpose this?
    # target = data.reshape(-1)  # replication # shape: [max_seq_len * batch_size]
    target = torch.fliplr(data).reshape(-1)  # reverse
    # target = source[i + 1:i + 1 + seq_len].reshape(-1)  # language modeling # shape: [max_seq_len * batch_size]
    return data, target

def initialize_model(model):
    # share embedding
    # self.dec_embedding.weight = self.src_tok_emb.weight

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def train(model: nn.Module) -> None:
    print("start training")
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 100
    start_time = time.time()

    num_batches = len(train_data) // cfg.max_seq_len
    print("num_batches:", num_batches)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, cfg.max_seq_len)):
        # print("batch", batch, "i", i)
        data, targets = get_batch(train_data, i)
        # truncate last batch
        if data.size(1) < cfg.max_seq_len:
            print("pass last batch")
            continue
        batch_size = data.size(0)
        output = model(data, data)  # TODO: why one input
        loss = criterion(output.view(-1, vocab_size), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    print("start testing")
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, cfg.max_seq_len):
            data, targets = get_batch(eval_data, i)
            # truncate last batch
            if data.size(1) < cfg.max_seq_len:
                print("pass last batch")
                continue
            batch_size = data.size(0)
            output = model(data, data)
            output_flat = output.view(-1, vocab_size)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("total # of params:", params)

if __name__ == "__main__":
    # TODO: copy
    # TODO: enc_input - src, dec_input - reverse src, target - shifted one reverse src (like translation task)
    # TODO: use IWSLT dataset or (try smaller PTB dataset)
    # train, valid, test = datasets.PennTreebank(root='./data')

    # Build vocab with WikiText2
    train_iter = WikiText2(split="train")
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        map(tokenizer, train_iter),  # apply tokenizer to train data to build vocab
        specials=["<unk>", "<pad>", "<bos>", "<eos>"],  # we can add <bos>, <eos>, but language model doesn't need them
        min_freq=5,  # control the size of vocab
    )
    vocab.set_default_index(
        vocab["<unk>"]
    )  # This index will be returned when OOV token is queried

    # check pad token id
    pad_idx = vocab["<pad>"]  # 1

    vocab_size = len(vocab)

    # Preprocess data - pad or cut the sequence
    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)  # shape: [2049990]
    val_data = data_process(val_iter)  # shape: [214417]
    test_data = data_process(test_iter)  # shape: [241859]

    # batchify data
    train_data = batchify(train_data, cfg.batch_size)  # shape: [64062, 32]
    val_data = batchify(val_data, cfg.eval_batch_size)  # shape: [13401, 16]
    test_data = batchify(test_data, cfg.eval_batch_size)  # shape: [15116, 16]

    # from torch.utils.data import DataLoader
    #
    # train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


    # build dataloader
    # we want input and target to be same
    # chunk the train data batch with batch size torch.chunk(train_data_batch, batch_size)
    # shuffle them - 1 epoch
    # shift input to the right and add <bos>, and add <eos> at the end

    # check if the model copy the unseen input
    # or reverse (change get_batch func)

    # define model
    print("device:", cfg.device)

    model = Transformer(n_layers=cfg.n_layers,
                        d_model=cfg.d_model,
                        h=cfg.h,
                        ffn_hidden=cfg.ffn_hidden,
                        p_drop=cfg.p_drop,
                        vocab_size=vocab_size,
                        max_seq_len=cfg.max_seq_len,
                        pad_idx=pad_idx,
                        device=cfg.device)

    model.to(cfg.device)
    print("parameter count:", count_parameters(model))

    # Training
    criterion = nn.CrossEntropyLoss()
    lr = 5.0  # learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


    # test forward
    test_src_input = get_batch(train_data, 0)[0].to(cfg.device)
    test_tgt_input = test_src_input

    forward_result = model(test_src_input, test_tgt_input)
    # test if output shape is correct
    assert forward_result.shape == (cfg.batch_size, cfg.max_seq_len, vocab_size)


    best_val_loss = float('inf')
    epochs = 3
    best_model = None

    # train model
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train(model)
        val_loss = evaluate(model, val_data)
        val_ppl = math.exp(val_loss)
        elapsed = time.time() - epoch_start_time
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

        scheduler.step()

    # test model
    test_loss = evaluate(best_model, test_data)
    test_ppl = math.exp(test_loss)
    print('=' * 89)
    print(f'| End of training | test loss {test_loss:5.2f} | '
          f'test ppl {test_ppl:8.2f}')
    print('=' * 89)

    # save model
    MODEL_PATH = './checkpoint/reverse.pth'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # 'loss': loss
    }, MODEL_PATH)

    breakpoint()
