import pandas as pd
import numpy as np


def load_dirty_data(in_path, ct_cutoff=10, out_path = None):
    """
    Load in the data and do some cleaning
    Specify out_path to save the data to a csv file
    """

    df = pd.read_csv(in_path, header=None, dtype = str)
    df.columns = ['Artist', 'Album', 'Track', 'Timestamp']

    #Lower case since the data is a bit messy
    df[['Artist', 'Track']] = df[['Artist', 'Track']].apply(lambda r: [x.lower() for x in r])

    # Parse the timestamps
    df['Timestamp'] = df['Timestamp'].apply(pd.to_datetime)
    df = df[df.Timestamp != pd.Timestamp('1970-01-01')]
    df = df.dropna(subset=['Timestamp'])

    df = df.sort_values('Timestamp')
    # Filter out tracks with too few plays
    #Group by both Artist and Track since there
    #are different tracks with the same name
    artist_track_grpby = df.groupby(['Artist', 'Track'], as_index = False, sort = False)
    cts = artist_track_grpby.size().reset_index()
    cts.columns = ['Artist', 'Track', 'ct']
    cts = cts[cts['ct'] > ct_cutoff]

    #Merge the counts in and drop tracks with too
    #few plays
    df = pd.merge(df, cts, on=['Artist', 'Track'], how = 'left').dropna(subset = ['ct'])

    # Give each track an id number
    ids = df.groupby(['Artist', 'Track'], as_index = False, sort = False).first()
    ids['track_id'] = range(len(ids))
    ids = ids[['Artist', 'Track', 'track_id']]
    df  = pd.merge(df, ids, on = ['Artist', 'Track'], how = 'left')

    #So we can figure out which id corresponds to which track
    id_lookup = ids[['Artist', 'Track']].values

    if out_path is not None:
        df.to_csv(out_path, index = False)

    return df, id_lookup, cts


def load_clean_data(path):
    """
    If the data has already been cleaned, we
    just need to reconstruct id_lookup and cts
    """

    df = pd.read_csv(path, dtype = {'ct':int, 'track_id':int})
    df['Timestamp'] = df['Timestamp'].apply(pd.to_datetime)


    unique_tracks = df.groupby(['Artist', 'Track'], as_index = False).first()

    cts = unique_tracks[['Artist', 'Track', 'ct']]
    id_lookup = unique_tracks.sort_values('track_id')[['Artist', 'Track']].values

    return df, id_lookup, cts
