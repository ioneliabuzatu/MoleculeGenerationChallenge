import numpy as np
import pandas as pd
import torch
import tqdm
from torch.autograd import Variable

from src.data import tensor_from_chars_list


def added_to_dictionary(smile, dictionary):
    for char in smile:
        if char not in dictionary:
            dictionary[char] = True


def to_numpy_compress(save=True):
    """Each smile is added an `!` and terminated with `space`. Each simbol in the smile in added to a dictionary
    :return
            dataset: the preprocessed smiles
            vocabulary: the set of all symbols that form the smiles
    """

    smiles = pd.read_csv("resources/data/smiles_train.txt", header=None)[0]

    dataset = []
    dictionary = {}

    for smile in smiles:
        smile = smile.strip()
        smile = '!' + smile + ' '
        dataset.append(smile)
        added_to_dictionary(smile, dictionary)

    vocabulary = np.array([ele for ele in dictionary], dtype=object)
    dataset = np.array(dataset, dtype=object)

    if save:
        np.savez_compressed('resources/data/smiles_data.npz', data_set=dataset, vocabulary=vocabulary)

    return dataset, vocabulary


def process_batch(sequences, batch_size, vocabulary):
    batches = []
    for i in range(0, len(sequences), batch_size):
        input_list = []
        output_list = []
        for j in range(i, i + batch_size, 1):
            if j < len(sequences):
                input_list.append(tensor_from_chars_list(sequences[j][:-1], vocabulary))
                output_list.append(tensor_from_chars_list(sequences[j][1:], vocabulary))
        inp = Variable(torch.cat(input_list, 0))
        target = Variable(torch.cat(output_list, 0))
        batches.append((inp, target))
    train_split = int(0.9 * len(batches))
    return batches[:train_split], batches[train_split:]


def save_preprocessed_data_as_batches(data, vocabulary, batch_size=128):
    hash_length_data = {}
    for smile in data:
        len_smile = len(smile)
        if len_smile >= 3:
            if len_smile not in hash_length_data:
                hash_length_data[len_smile] = []
            hash_length_data[len_smile].append(smile)
    train_batches = []
    val_batches = []
    for length in tqdm.tqdm(hash_length_data):
        train, val = process_batch(hash_length_data[length], batch_size, vocabulary)
        train_batches.extend(train)
        val_batches.extend(val)
    torch.save({"train": train_batches, "val": val_batches}, "resources/data/train_val_batches.npz")
    print(f"saved train and val batches to resources/data/train_val_batches.npz")


if __name__ == "__main__":
    data, symbols = to_numpy_compress(save=True)
    save_preprocessed_data_as_batches(data, symbols)
