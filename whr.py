# implementation of Bradley-Terry Whole-History Rating (Maximum Likelihood Estimation for Elo Ratings)
# CPUs: model A | model B | winner
# AI vs. Humans: model | Human username | winner - consider just using Smogon Elo for the ranked ladder calculations!
# some code borrowed from https://colab.research.google.com/drive/1KdwokPjirkTmpO_P1WByFNFiqxWQquwH#scrollTo=mSizG3Pzglte

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

def whr(df: pd.DataFrame, SCALE: int=400, BASE: int=10, INIT_RATING: int=1000):
    models = pd.concat([df['model_a'], df['model_b']]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    print(df)
    p = len(models)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +np.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -np.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    return pd.Series(elo_scores, index = models.index).sort_values(ascending=False)

if __name__ == '__main__':
    file = 'battle_log/gen9ou.csv'
    df = pd.read_csv(file + '.csv')
    df_new = whr(df)
    df_new.to_csv(file + 'results.csv', mode='w')