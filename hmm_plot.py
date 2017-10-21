from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category20
from bokeh.embed import components
import numpy as np


def plot_most_likely_state(model, X, timestamps):
    state_probs = model.predict_proba(X)
    most_likely_states = state_probs.argmax(1)
    map_prob = state_probs.max(1)

    output_file("hmm_predictions_chart.html", title="HMM Predictions")

    p = figure(x_axis_type = "datetime")

    for x in xrange(model.n_components):
        x_coords = timestamps.iloc[np.where(most_likely_states == x)[0]]
        y_coords = map_prob[most_likely_states == x]
        p.circle(x_coords, y_coords,
                 fill_color=Category20[model.n_components][x],
                 fill_alpha=.6, line_color=None, legend=str(x))

    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    p.xaxis.axis_label = "State Probability"
    p.yaxis.axis_label = "Date"
    script, div = components(p)
    show(p)
    return script, div
