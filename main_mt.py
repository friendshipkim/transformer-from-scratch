from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import IWSLT2016, Multi30k
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from timeit import default_timer as timer

from typing import Iterable, List

import torch
import torch.nn as nn
from tqdm import tqdm
import copy

from model.transformer import Transformer
from baseline_model.baseline_transformer import BaselineTransformer
import config as cfg
from tests.test_utils import *

import numpy as np
import random

import argparse

SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"

# fix random seed
torch.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(cfg.seed)
torch.cuda.manual_seed_all(cfg.seed)  # if use multi-GPU
np.random.seed(cfg.seed)
random.seed(cfg.seed)
os.environ["PYTHONHASHSEED"] = str(cfg.seed)


def weight_init_baseline(model):
    print("initializing baseline model")
    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)


def weight_init_my(model):
    print("initializing my model")
    # make bias zero
    zero_biases = [f"{enc_dec}.layers.{layer_num}.{attn}.{proj}.bias"
                   for enc_dec in ["encoder", "decoder"]
                   for layer_num in range(cfg.n_layers)
                   for attn in ["self_attn", "cross_attn"]
                   for proj in ["fc_q", "fc_k", "fc_v", "fc_concat"]]

    # initialize
    for name, param in transformer.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        elif name in zero_biases:
            zero_biases.remove(name)
            nn.init.zeros_(param)
        else:
            pass
    assert len(zero_biases) == 4 * cfg.n_layers, "bias not initialized correctly"


def train_epoch(model, optimizer):
    model.train()
    losses = 0

    train_iter = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=cfg.batch_size, collate_fn=collate_fn)

    for src, tgt in tqdm(train_dataloader):
        src = src.t().to(cfg.device)
        tgt = tgt.t().to(cfg.device)

        tgt_input = tgt[:, :-1]  # batch_first

        _, _, _, _, logits = model(src, tgt_input)

        optimizer.zero_grad()

        tgt_out = tgt[:, 1:]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split="valid", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=cfg.batch_size, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.t().to(cfg.device)
        tgt = tgt.t().to(cfg.device)

        tgt_input = tgt[:, :-1]

        _, _, _, _, logits = model(src, tgt_input)

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


def test(model):
    model.eval()
    losses = 0

    test_iter = Multi30k(split="test", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    test_dataloader = DataLoader(test_iter, batch_size=cfg.batch_size, collate_fn=collate_fn)

    for src, tgt in test_dataloader:
        src = src.t().to(cfg.device)
        tgt = tgt.t().to(cfg.device)

        tgt_input = tgt[:, :-1]

        _, _, _, _, logits = model(src, tgt_input)

        tgt_out = tgt[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(test_dataloader)


if __name__ == "__main__":
    # settings
    parser = argparse.ArgumentParser(description="Machine Translation Example")
    parser.add_argument("--model-type", action="store", default="baseline",
                        help="model type: baseline | my_copy | my")
    parser.add_argument("--dataset", action="store", default="Multi30K",
                        help="dataset: Multi30K | IWSLT2016")
    parser.add_argument("--evaluate", action="store_true", default=False,
                        help="whether to test the model")

    args = parser.parse_args()
    print(vars(args))

    # check options
    if not (args.model_type in ["baseline", "my", "my_copy"]):
        print("ERROR: Please choose the correct model type")
        exit()

    # model path
    model_filename = f"{args.model_type}.{cfg.n_layers}-layers.{cfg.h}-heads.pth"
    MODEL_PATH = os.path.join("./checkpoint/", model_filename)

    """
    build vocab
    """
    # Placeholders
    token_transform = {}
    vocab_transform = {}

    # Create source and target language tokenizer. Make sure to install the dependencies.
    # pip install -U spacy
    # python -m spacy download en_core_web_sm
    # python -m spacy download de_core_news_sm
    token_transform[SRC_LANGUAGE] = get_tokenizer("spacy", language="de_core_news_sm")
    token_transform[TGT_LANGUAGE] = get_tokenizer("spacy", language="en_core_web_sm")

    # helper function to yield list of tokens
    def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
        language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

        for data_sample in data_iter:
            # yield tokenized sentence(list) of given language
            yield token_transform[language](data_sample[language_index[language]])


    # Define special symbols and indices
    UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
    # Make sure the tokens are in order of their indices to properly insert them in vocab
    special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        # Training data Iterator
        train_iter = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        # Create torchtext's Vocab object
        vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                        min_freq=1,
                                                        specials=special_symbols,
                                                        special_first=True)

    # Set UNK_IDX as the default index. This index is returned when the token is not found.
    # If not set, it throws RuntimeError when the queried token is not found in the Vocabulary.
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        vocab_transform[ln].set_default_index(UNK_IDX)

    """
    define models
    """
    SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
    TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

    if args.model_type == "my":
        transformer = Transformer(n_layers=cfg.n_layers,
                                  d_model=cfg.d_model,
                                  h=cfg.h,
                                  ffn_hidden=cfg.ffn_hidden,
                                  p_drop=cfg.p_drop,
                                  src_vocab_size=SRC_VOCAB_SIZE,
                                  tgt_vocab_size=TGT_VOCAB_SIZE,
                                  pad_idx=PAD_IDX,
                                  device=cfg.device)

        transformer.to(cfg.device)
        # weight_init_baseline(transformer)

    else:
        b_transformer = BaselineTransformer(num_encoder_layers=cfg.n_layers,
                                            num_decoder_layers=cfg.n_layers,
                                            emb_size=cfg.d_model,
                                            nhead=cfg.h,
                                            src_vocab_size=SRC_VOCAB_SIZE,
                                            tgt_vocab_size=TGT_VOCAB_SIZE,
                                            dim_feedforward=cfg.ffn_hidden,
                                            dropout=cfg.p_drop,
                                            pad_idx=PAD_IDX,
                                            device=cfg.device)
        b_transformer.to(cfg.device)
        # weight_init_baseline(b_transformer)

        if args.model_type == "baseline":
            transformer = b_transformer
        elif args.model_type == "my_copy":  # my copy model
            transformer = Transformer(n_layers=cfg.n_layers,
                                      d_model=cfg.d_model,
                                      h=cfg.h,
                                      ffn_hidden=cfg.ffn_hidden,
                                      p_drop=cfg.p_drop,
                                      src_vocab_size=SRC_VOCAB_SIZE,
                                      tgt_vocab_size=TGT_VOCAB_SIZE,
                                      pad_idx=PAD_IDX,
                                      device=cfg.device)

            transformer.to(cfg.device)

            # copy transformer weights
            copied_my_sd = copy_transformer_dict(src=b_transformer.state_dict(),
                                                 tgt=copy.deepcopy(transformer.state_dict()),
                                                 n_layers=cfg.n_layers)
            transformer.load_state_dict(copied_my_sd)


    """
    loss & optimizer
    """
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    # optimizer = torch.optim.SGD(transformer.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=cfg.lr, betas=(0.9, 0.98), eps=1e-9)


    """
    collation
    """
    # helper function to club together sequential operations
    def sequential_transforms(*transforms):
        def func(txt_input):
            for transform in transforms:
                txt_input = transform(txt_input)
            return txt_input
        return func

    # function to add BOS/EOS and create tensor for input sequence indices
    def tensor_transform(token_ids: List[int]):
        return torch.cat((torch.tensor([BOS_IDX]),
                          torch.tensor(token_ids),
                          torch.tensor([EOS_IDX])))

    # src and tgt language text transforms to convert raw strings into tensors indices
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln],  # Tokenization
                                                   vocab_transform[ln],  # Numericalization
                                                   tensor_transform)  # Add BOS/EOS and create tensor

    # function to collate data samples into batch tensors
    def collate_fn(batch):
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
            tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

        src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
        tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
        return src_batch, tgt_batch


    """
    Test model
    """
    if args.evaluate:
        if os.path.exists(MODEL_PATH):
            print("Evaluate pretrained model")
            checkpoint = torch.load(MODEL_PATH)
            transformer.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            val_loss = checkpoint['val_loss']

            test_loss = test(transformer)
            print(f"Test loss: {test_loss:.4f}")
            exit()
        else:
            assert False, f"'{MODEL_PATH}' doesn't exist"

    """
    Let's train!
    """
    best_val_loss = float('inf')
    for epoch in range(1, cfg.epochs + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        print(
            f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "f"Epoch time = {(end_time - start_time):.3f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss
            }, MODEL_PATH)
            print(f"model saved to '{MODEL_PATH}'")

