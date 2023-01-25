import math

import numpy as np
import pandas as pd
import torch
from scipy.io import arff
from sklearn.preprocessing import RobustScaler

from utils import set_seed


def load_spam_data(seed, features_order=None):
    set_seed(seed)
    path = "./datasets/journalApontadorElsevier.arff"
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    most_disc = ['qTips_plc', 'rating_plc', 'qEmail_tip', 'qContacts_tip', 'qURL_tip', 'qPhone_tip', 'qNumeriChar_tip', 'sentistrength_tip', 'combined_tip', 'qWords_tip',
                 'followers_followees_gph', 'qUnigram_avg_tip', 'qTips_usr', 'indeg_gph', 'qCapitalChar_tip', 'class1']

    df = df[most_disc]
    df["class1"].replace({b'spam': -1, b'notspam': 1}, inplace=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    Y = df['class1'].values
    X = df.drop('class1', axis=1).values

    x_dim = len(X[0])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X /= math.sqrt(x_dim)
    # change column order
    if features_order:
        X = X[:, features_order]
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()


def load_card_fraud_data(seed, features_order=None):
    set_seed(0)
    df = pd.read_csv('./datasets/creditcard.csv')

    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    scaled_amount = df['scaled_amount']
    df.drop(['scaled_amount'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)

    df["Class"].replace({1: -1, 0: 1}, inplace=True)

    # Shuffle dataframe rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    Y = df['Class'].values
    X = df.drop('Class', axis=1).values
    x_dim = len(X[0])
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    X /= math.sqrt(x_dim)
    # change column order
    if features_order:
        X = X[:, features_order]
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()
