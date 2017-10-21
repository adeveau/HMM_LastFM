import numpy as np
import pandas as pd

class AugmentedModel(object):
    """
    A class to keep everything organized.
    Useful if you're trying different hyperparameters.
    analysis.py just calls the functions
    one by one since it's an example and easier to follow.
    """

    def __init__(self, model, id_lookup, steal_attributes = False):

        self.model = model
        self.id_lookup = id_lookup

        #If you want to be able to reference the attributes of model
        #directly. Generally inadvisable.

        if steal_attributes:
            for attribute in model.__dict__:
                self.__dict__[attribute] = model.__dict__[attribute]

        self.steady_state = steady_state(self.model)
        self.e_probs = process_eprobs(self.model.emissionprob_, self.id_lookup)
        self.cond_probs = build_cond_prob_df(self.e_probs)

    def id_lookup(id):
        return self.id_lookup[id]

    def top_e_probs(self, k):
        return top_k(self.e_probs, k)

    def bottom_e_probs(self, k):
        return bottom_k(self.e_probs, k)

    def top_cond_probs(self, k):
        return top_k(self.cond_probs, k)

    def bottom_cond_probs(self, k):
        return bottom_k(self.cond_probs, k)


def process_eprobs(emission_probs, id_lookup):
    """
    Turn the ids back into track names
    and make a DataFrame
    """
    return pd.DataFrame(emission_probs.T, index=id_lookup)


def calc_conditional_prob(song, state, emission_probs, steady_state):
    """
    Given that we observe a particular song, calculate
    the probability that we are in a particular state.
    """

    #Bayes' Rule
    numer = (emission_probs.loc[song, state] * steady_state[state])
    denom = np.dot(emission_probs.loc[song, :], steady_state)
    return numer / denom


def build_cond_prob_df(emission_probs, steady_state):
    """
    Calculate the conditional probabilities
    for all (song, state) pairs
    """

    cond_probs = []
    for x in xrange(emission_probs.shape[1]):
        cond_probs.append([calc_conditional_prob(
            song, x, emission_probs, steady_state) for song in emission_probs.index])

    return pd.DataFrame(np.array(cond_probs).T, index=emission_probs.index)

def steady_state(model):
    """
    Compute the steady state distribution of
    the model
    """
    e_vals, e_vecs = np.linalg.eig(model.transmat_)

    #The eigenvalue might not be exactly 1. due to
    #roundoff error, so take the closest
    return e_vecs[:, np.absolute(e_vals - 1).argmin()]


def top_k(df, k = 5):
    """
    Return a dataframe where each column
    contains the indices of the k largest
    entries of the corresponding column in df
    """

    top_k = []
    for col in df.columns:
        cur_col = df[col]

        #Quickselect. A bit faster than using a heap
        top_indices = np.argpartition(cur_col,-k)[-k:]
        top_k_col = cur_col.iloc[top_indices].sort_values(ascending = False).index
        top_k.append(top_k_col)

    return pd.DataFrame(np.array(top_k).T)

def bottom_k(df, k = 5):
    """
    Return a dataframe where each column
    contains the indices of the k smallest
    entries of the corresponding column in df
    """
    bottom_k = []
    for col in df.columns:
        #Quickselect. A bit faster than using a heap
        bottom_indices = np.argpartition(df[col],k)[:k]
        bottom_k_col = df[col].iloc[bottom_indices].sort_values(ascending = True).index
        bottom_k.append(bottom_k_col)

    return pd.DataFrame(np.array(bottom_k).T)
