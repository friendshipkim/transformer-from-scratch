import config  # hyperparameters
from model.transformer import Transformer

import torchtext.datasets
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import dataset
import torch
from torch import nn, Tensor


if __name__ == "__main__":
    # use IWSLT dataset
    # try smaller PTB dataset
    # train, valid, test = datasets.PennTreebank(root='./data')

    # preprocessing

    # padding or cut the sequence
    # build dataloader

    # WikiText2
    train_iter = WikiText2(split="train")
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        map(
            tokenizer, train_iter
        ),  # TODO: what does it mean? map(tokenizer, train_iter)
        specials=["<unk>", "<bos>", "<eos>"],
    )  # we can add <bos>, <eos>
    vocab.set_default_index(
        vocab["<unk>"]
    )  # This index will be returned when OOV token is queried

    def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
        """Converts raw text into a flat Tensor."""
        data = [
            torch.tensor(vocab(tokenizer(item)), dtype=torch.long)
            for item in raw_text_iter
        ]
        # vocab method maps tokens to indices.
        return torch.cat(
            tuple(filter(lambda t: t.numel() > 0, data))
        )  # TODO: why tuple?

    # train_iter was "consumed" by the process of building the vocab,
    # so we have to create it again
    train_iter, val_iter, test_iter = WikiText2()
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def batchify(data: Tensor, bsz: int) -> Tensor:
        """Divides the data into bsz separate sequences, removing extra elements
        that wouldn't cleanly fit.

        Args:
            data: Tensor, shape [N]
            bsz: int, batch size

        Returns:
            Tensor of shape [N // bsz, bsz]
        """
        seq_len = data.size(0) // bsz
        data = data[: seq_len * bsz]
        data = data.view(bsz, seq_len).t().contiguous()
        return data.to(device)

    batch_size = 20
    eval_batch_size = 10
    train_data_batch = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
    val_data_batch = batchify(val_data, eval_batch_size)
    test_data_batch = batchify(test_data, eval_batch_size)

    breakpoint()

    # build dataloader
    # we want input and target to be same
    # chunk the train data batch with batch size torch.chunk(train_data_batch, batch_size)
    # shuffle them - 1 epoch
    # shift input to the right and add <bos>, and add <eos> at the end

    # check if the model copy the unseen input
    # or reverse (change get_batch func)

    model = Transformer(
        n_layers=config.n_layers,
        d_model=config.d_model,
        h=config.h,
        ffn_hidden=config.ffn_hidden,
        p_drop=config.p_drop,
        d_embed=config.d_embed,
        vocab=vocab,
    )
