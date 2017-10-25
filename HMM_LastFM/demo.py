import pandas as pd
import numpy as np
import hmm_plot
from load_data import load_clean_data
from hmmlearn import hmm
from hmm_helpers import AugmentedModel
from sklearn.externals import joblib

#Load our data and set the hyperparameters of the model
data, id_lookup, play_counts = load_clean_data("adeveau9_10_20_2017_clean.csv")
amodel = AugmentedModel(8, id_lookup)

#Fit the model and save a serialized version
X = data['track_id'].values.reshape(-1, 1)
amodel.fit(X)
joblib.dump(amodel, "trained_model.pkl")

#Determine the steady state of our model
amodel.steady_state

#A dataframe of the emissions probabilities
amodel.e_probs_df.head(10)

#Check the most and least likely tracks to be emitted from each state
most_likely = amodel.top_e_probs(5)
least_likely = amodel.bottom_e_probs(5)

#What if we want to know about p(state|song)? A dataframe
#with the conditional probabilities
amodel.cond_probs.head(10)
top_cond_probs = amodel.top_cond_probs(5)

#Make some plots. They're embedded in the post, so
#we'll want to get the script and <div> tag used by bokeh
script_bars, div_bars = hmm_plot.plot_bars(play_counts)
script_states, div_stats = hmm_plot.plot_most_likely_state(amodel, X, data['Timestamp'])
