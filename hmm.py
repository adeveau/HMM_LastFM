from hmmlearn import hmm
import pandas as pd
import sys
import numpy as np


def load_data(artist=None, ct_cutoff=10):
    """
    Load in the data and do some preprocessing
    """

    df = pd.read_csv("adeveau9_lastfm_10_20_2017.csv", header=None)[::-1]
    df.columns = ['Artist', 'Album', 'Track', 'Timestamp']

    if artist is not None:
        df = df[df['Artist' == artist]]

    # Parse the timestamps
    df['Timestamp'] = df['Timestamp'].apply(pd.to_datetime)
    df = df[df.Timestamp != pd.Timestamp('1970-01-01')]
    df = df.dropna(subset=['Timestamp'])

    # Filter out tracks with too few plays
    cts = df['Track'].value_counts().reset_index()
    cts.columns = ['Track', 'ct']
    cts = cts[cts['ct'] > ct_cutoff]
    df = df[df['Track'].isin(cts['Track'])]

    # Give each track an id number
    ids = {}
    id_lookup = []
    for i, track in enumerate(df['Track'].unique()):
        ids[track] = i
        id_lookup.append(track)

    df['track_id'] = df['Track'].apply(lambda x: ids[x])
    df.sort_values('Timestamp')
    #df['TimeToNextPlay'] = df['Timestamp'].shift(-1) - df['Timestamp']

    return df, id_lookup


def process_eprobs(emission_probs, id_lookup):
    return pd.DataFrame(emission_probs.T, index=id_lookup)


def calc_conditional_prob(song, state, emission_probs, steady_state):
    """
    Given that we observe a given song, calculate
    the probability that we are in a given state.
    """

    numer = (emission_probs.loc[song, state] * steady_state[state])
    denom = np.dot(emission_probs.loc[song, :], steady_state)
    return numer / denom


def build_cond_prob_df(emission_probs, steady_state):
    """
Calculate the conditional probabilities
    for all (song, state) pairs
    """
    cond_probs = []
    for x in xrange(model.n_components):
        cond_probs.append([calc_conditional_prob(
            song, x, emission_probs, steady_state) for song in emission_probs.index])

    return pd.DataFrame(np.array(cond_probs).T, index=emission_probs.index)


def steady_state(model):
    """
    Compute the steady state distribution of
    the model
    """
    e_vals, e_vecs = np.linalg.eig(model.transmat_)
    return e_vecs[:, np.absolute(e_vals - 1).argmin()]


# Visualize most likely state over time

# Check which songs get the biggest information gain
# by conditioning on a group

# Display things with the highest emission probability


if __name__ == "__main__":
    data, id_lookup = load_data()
    model = hmm.MultinomialHMM(n_components=8)

    X = data['track_id'].values.reshape(-1, 1)
    model.fit(X)

    eprobs_df = process_eprobs(model.emissionprob_, id_lookup)
