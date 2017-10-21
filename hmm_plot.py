import numpy as np
from bokeh.plotting import figure, show, output_file
from bokeh.palettes import Category20
from bokeh.embed import components

def plot_bars(cts):
    """
    Plot a simple chart showing
    how many times I've listened
    to each track in the dataset
    """

    cts = cts.sort_values('ct')
    output_file("cts_bar.html", title = "Counts Bar Chart")

    #Set up the plot
    p = figure(x_range = cts['Track'].values)
    p.xaxis.visible = False
    p.yaxis.axis_label = "# of Plays"
    p.vbar(x = cts['Track'].values, top = cts['ct'].values, width = .9)

    show(p)
    #Return a script and a <div> tag for embedding
    return components(p)

def plot_most_likely_state(model, X, timestamps):
    """
    Plot a scatter plot showing what the model
    has inferred to be the most likely state sequence
    and the probability of being in the most likely state
    """

    #Compute the state probabilities
    #And get the most likely state along
    #with the probability of being in that state
    state_probs = model.predict_proba(X)
    most_likely_states = state_probs.argmax(1)
    map_prob = state_probs.max(1)

    #Set up our figure
    output_file("hmm_predictions_chart.html", title="HMM State Predictions")

    p = figure(x_axis_type = "datetime")
    p.legend.location = 'top_left'
    p.legend.click_policy = 'hide'
    p.xaxis.axis_label = "Date"
    p.yaxis.axis_label = "State Probability"

    #Plot each state in a different color and add it to the legend
    for x in xrange(model.n_components):
        x_coords = timestamps.iloc[np.where(most_likely_states == x)[0]]
        y_coords = map_prob[np.where(most_likely_states == x)]
        p.circle(x_coords, y_coords,
                 fill_color=Category20[model.n_components][x],
                 fill_alpha=.6, line_color=None, legend=str(x))

    show(p)
    #Return a script and a <div> tag for embedding
    return components(p)
