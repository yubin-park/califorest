import numpy as np
import pandas as pd
import sklearn.metrics as skm
import tqdm

def hosmer_lemeshow(yTrue, yScore):
    """
    Calculate the Hosmer Lemeshow to assess whether
    or not the observed event rates match expected
    event rates.

    Assume that there are 10 groups:
    HL = \\sum_{g=1}^G \\frac{(O_{1g} - E_{1g})^2}{N_g \\pi_g (1- \\pi_g)}
    """
    # create the dataframe
    scoreDF = pd.DataFrame({'score': yScore, 'target': yTrue})
    # sort the values
    scoreDF = scoreDF.sort_values('score')
    # shift the score a bit
    scoreDF['score'] = np.clip(scoreDF['score'], 1e-16, 1-1e-16)
    scoreDF['rank'] = list(range(scoreDF.shape[0]))
    # cut them into 10 bins
    scoreDF['score_decile'] = pd.qcut(scoreDF['rank'], 10,
                                      duplicates='raise')
    # sum up based on each decile
    obsPos = scoreDF['target'].groupby(scoreDF.score_decile).sum()
    obsNeg = scoreDF['score'].groupby(scoreDF.score_decile).count() - obsPos
    exPos = scoreDF['score'].groupby(scoreDF.score_decile).sum()
    exNeg = scoreDF['score'].groupby(scoreDF.score_decile).count() - exPos
    hl = (((obsPos - exPos)**2/exPos) + ((obsNeg - exNeg)**2/exNeg)).sum()
    return hl


def spiegelhalter(yTrue, yScore):
    top = np.sum((yTrue - yScore)*(1-2*yScore))
    bot = np.sum((1-2*yScore)**2 * yScore * (1-yScore))
    return top / np.sqrt(bot)

def scaled_Brier(yTrue, yScore):
    brier = skm.brier_score_loss(yTrue, yScore)
    # calculate the mean of the probability
    p = np.mean(yTrue)  
    sBrier = 1- brier / (p * (1-p))
    return brier, sBrier
