import argparse
import pickle

import numpy as np
import pandas as pd
from fcd import canonical_smiles, get_predictions, load_ref_model, calculate_frechet_distance

parser = argparse.ArgumentParser()
parser.add_argument("--submission", type=str, default="resources/my_smiles.txt")
parser.add_argument("--teststats", type=str, default="resources/test_stats.p")
parser.add_argument("--traindata", type=str, default="resources/data/smiles_train.txt")
args = parser.parse_args()


def get_metrics():
    model = load_ref_model()

    smiles_gen = pd.read_csv(args.submission, header=None)[0][:10000]
    smiles_gen_can = [w for w in canonical_smiles(smiles_gen) if w is not None]

    act_gen = get_predictions(model, smiles_gen_can)
    mu_gen = np.mean(act_gen, axis=0)
    sigma_gen = np.cov(act_gen.T)

    with open(args.teststats, 'rb') as f:
        mu_test, sigma_test = pickle.load(f)

    fcd_value = calculate_frechet_distance(
        mu1=mu_gen,
        mu2=mu_test,
        sigma1=sigma_gen,
        sigma2=sigma_test)
    print(f"FCD: {fcd_value:.3f}")

    validity = len(smiles_gen_can) / len(smiles_gen)
    print("Validity: ", validity)

    smiles_unique = set(smiles_gen_can)
    uniqueness = len(smiles_unique) / len(smiles_gen)
    print("Uniqueness: ", uniqueness)

    with open(args.traindata) as f:
        smiles_train = {s for s in f.read().split() if s}

    smiles_novel = smiles_unique - smiles_train
    novelty = len(smiles_novel) / len(smiles_gen)
    print("Novelty: ", novelty)


if __name__ == '__main__':
    get_metrics()