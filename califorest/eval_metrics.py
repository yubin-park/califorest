import numpy as np
import pandas as pd
import sklearn.metrics as skm
import tqdm
from scipy.stats import chi2
from scipy.stats import norm

def hosmer_lemeshow(y_true, y_score):
    """
    Calculate the Hosmer Lemeshow to assess whether
    or not the observed event rates match expected
    event rates.

    Assume that there are 10 groups:
    HL = \\sum_{g=1}^G \\frac{(O_{1g} - E_{1g})^2}{N_g \\pi_g (1- \\pi_g)}
    """
    n_grp = 10 # number of groups

    # create the dataframe
    df = pd.DataFrame({'score': y_score, 'target': y_true})

    # sort the values
    df = df.sort_values('score')
    # shift the score a bit
    df['score'] = np.clip(df['score'], 1e-8, 1-1e-8)
    df['rank'] = list(range(df.shape[0]))
    # cut them into 10 bins
    df['score_decile'] = pd.qcut(df['rank'], n_grp,
                                      duplicates='raise')
    # sum up based on each decile
    obsPos = df['target'].groupby(df.score_decile).sum()
    obsNeg = (df['target'].groupby(df.score_decile).count() - 
                obsPos)
    exPos = df['score'].groupby(df.score_decile).sum()
    exNeg = df['score'].groupby(df.score_decile).count() - exPos
    hl = (((obsPos - exPos)**2/exPos) + ((obsNeg - exNeg)**2/exNeg)).sum()

    # https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test
    # Re: p-value, higher the better Goodness-of-Fit
    p_value = 1 - chi2.cdf(hl, n_grp-2)
    
    return p_value

def reliability(y_true, y_score):

    n_grp = 10
    df = pd.DataFrame({'score': y_score, 'target': y_true})
    df = df.sort_values('score')
    df['rank'] = list(range(df.shape[0]))
    df['score_decile'] = pd.qcut(df['rank'], n_grp,
                                      duplicates='raise')

    obs = df['target'].groupby(df.score_decile).mean()
    exp = df['score'].groupby(df.score_decile).mean()

    rel_small = np.mean((obs - exp)**2)
    rel_large = (np.mean(y_true) - np.mean(y_score))**2

    return rel_small, rel_large

def spiegelhalter(y_true, y_score):
    top = np.sum((y_true - y_score)*(1-2*y_score))
    bot = np.sum((1-2*y_score)**2 * y_score * (1-y_score))
    sh = top / np.sqrt(bot)

    # https://en.wikipedia.org/wiki/Z-test
    # Two-tailed test
    # Re: p-value, higher the better Goodness-of-Fit
    p_value = norm.sf(np.abs(sh)) * 2

    return p_value

def scaled_Brier(y_true, y_score):
    brier = skm.brier_score_loss(y_true, y_score)
    # calculate the mean of the probability
    p = np.mean(y_true)  
    brier_scaled = 1 - brier / (p * (1-p))
    return brier, brier_scaled
