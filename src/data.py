import torch

import config


def load_train_and_val_batches_data():
    if not config.host:
        f = torch.load("utils/resources/data/train_val_batches.npz")
    else:
        f = torch.load("~/data/smiles-project/train_val_batches.npz")
    return f['train'], f['val']


def max_length_smile():
    max_len_smiles = 0
    with open("utils/resources/sample_submission.txt", "r") as file:
        content_file = file.readlines()
        for smile in content_file:
            smile = smile.strip()
            len_smile = len(smile)
            max_len_smiles = max(max_len_smiles, len_smile)

    print("Longest smile found has length", max_len_smiles)


def tensor_from_chars_list(chars_list, vocabs):
    tensor = torch.zeros(len(chars_list)).long()
    for c in range(len(chars_list)):
        tensor[c] = vocabs.index(chars_list[c])
    return tensor.view(1, -1)
