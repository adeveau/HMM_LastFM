import pandas as pd
import numpy as np
from load_data import load_data
from hmmlearn import hmm
import hmm_helpers as helpers
import hmm_plot
from sklearn.externals import joblib

#Load our data and set the hyperparameters of the model
data, id_lookup, play_counts = load_data("adeveau9_lastfm_10_20_2017.csv")
model = hmm.MultinomialHMM(n_components=8)

#Fit the model and save a serialized version
model.fit(data['track_id'].values.reshape(-1, 1))
joblib.dump(model, "trained_model.pkl")

#Determine the steady state of our model
steady_state_probs = helpers.steady_state(model)

#Build a dataframe of the emissions probabilities
e_probs = helpers.process_eprobs(model.emissionprob_, id_lookup)

#Check the most and least likely tracks to be emitted from each state
most_likely = helpers.top_k(e_probs, k = 5)
least_likely = helpers.bottom_k(e_probs, k = 5)

#What if we want to know about p(state|song)? Build a dataframe
#with the conditional probabilities
cond_probs = helpers.build_cond_prob_df(e_probs, steady_state_probs)
top_cond_probs = helpers.top_k(cond_probs, k = 5)


#Make some plots. They're embedded in the post, so
#we'll want to get the script and <div> tag used by bokeh
script_bars, div_bars = hmm_plot.plot_bars(play_counts)
script_states, div_stats = hmm_plot.plot_most_likely_state(model, X, )
