import pandas as pd
import numpy as np


def load_data(path, artist=None, ct_cutoff=10):
    """
    Load in the data and do some preprocessing
    """

    df = pd.read_csv(path, header=None)[::-1]
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

    return df, id_lookup, cts
