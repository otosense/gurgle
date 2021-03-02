"""
To run, run this in the terminal:

```
streamlit run streamlit_app.py
```
"""
import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

st.title('My first app')

# st.write("Here's our first attempt at using data to create a table:")
# st.write(pd.DataFrame({
#     'first column': [1, 2, 3, 4],
#     'second column': [10, 20, 30, 40]
# }))
#
# df = pd.DataFrame({
#     'another col': ['apples', 'and', 'bananas'],
#     'and yet another': [[1, 2], 'yum', float]
# })
#
# df  # lone litteral will result in a st.write call
#
# df.T;

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import time

# max_x = 5
# max_y = 10

from itertools import cycle
from collections import deque

# def cyclic_data_gen(max_items=100):
#     c = cycle(range(max_y))
#     d = deque(range(max_y), max_x)
#     for i in range(max_items):
#         d.append(next(c))
#         yield d

from itertools import tee, count, chain, islice


def cyclic_data_gen(max_items=100, max_y=10):
    yield from islice(cycle(range(max_y)), 0, max_items)


def prepend_several(iterator, fillval=None, n_prepends=1):
    """Prepend a single value in front of an iterator
    >>> list(prepend_several([2, 3, 4], 0, n_prepends=3))
    [0, 0, 0, 2, 3, 4]
    """
    "Prepend a single value in front of an iterator"
    return chain([fillval] * n_prepends, iterator)


def sliding_window(iterable, n=2):
    """Iterable of sliding windows
    >>> list(sliding_window(range(7), 3))
    [(0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6)]
    """
    iterables = tee(iterable, n)

    for iterable, num_skipped in zip(iterables, count()):
        for _ in range(num_skipped):
            next(iterable, None)

    return zip(*iterables)


def streaming_graph(data_gen, window_length=5, inter_tick_sleep=0.00, max_y=10):
    sliding_window_intervals = sliding_window(prepend_several(data_gen, 0, window_length - 1), window_length)

    # sliding_window_intervals = sliding_window(data_gen, window_length)
    fig, ax = plt.subplots()

    x = np.arange(0, window_length)
    ax.set_ylim(0, max_y)
    line, = ax.plot(x, next(sliding_window_intervals), '-o')
    the_plot = st.pyplot(plt)

    def init():  # give a clean slate to start
        line.set_ydata([np.nan] * len(x))

    init()

    for i, interval in enumerate(sliding_window_intervals):
        line.set_ydata(interval)
        the_plot.pyplot(plt)
        if inter_tick_sleep:
            time.sleep(inter_tick_sleep)


from taped import LiveWf, WfChunks

with WfChunks() as wf_chunks:
    def gen():
        for chk in islice(wf_chunks, 0, 200):
            fv = np.std(chk)
            # print(f"{fv:0.0f}")
            print(f"{int(fv):b}")
            yield fv

    print('------ begin!!!')
    # streaming_graph(gen(), window_length=10, max_y=800)
    for x in gen():
        pass

    # streaming_graph(cyclic_data_gen(20, max_y=10))

#
# chart_data = pd.DataFrame(
#     np.random.randn(20, 3),
#     columns=['a', 'b', 'c'])
#
# st.line_chart(chart_data)
#
# import streamlit as st
# import pandas as pd
# import time
# import altair as alt
# from altair import Chart, X, Y, Axis, SortField, OpacityValue
#
#
# # ---------------------------------------------------------------#
# # Creating an empty chart in the beginning when the page loads
# # ---------------------------------------------------------------#
# bars = alt.Chart(data).mark_bar().encode(
#     x=X('1:Q', axis=Axis(title='ATP Ranking Points'),
#         y=Y('0:Q', axis=Axis(title='The Big Four'))
#         ).properties(
#         width=650,
#         height=400
#     ))
# # This global variable 'bar_plot' will be used later on
# bar_plot = st.altair_chart(bars)
#
#
# x = st.slider('Select the year range',1996, 2017, (1996, 2017))

# def plot_bar_animated_altair(df, week):
#     bars = alt.Chart(df, title="Ranking as of week :" + week).encode(
#         x=X('ranking_points:Q', axis=Axis(title='ATP Ranking Points'),
#             y=Y('full_name:N', axis=Axis(title='The Big Four'), sort='-x'),
#             color=alt.Color('full_name:N'),
#             .properties(
#             width=650,
#             height=400
#         )))
#
#     if st.button('Cue Chart'):
#         for week in week_list:
#     # weekly_df -> this dataframe (sample shown above) contains
#     # data for a particular week which is passed to
#     # the 'plot_bar_animated_altair' function.
#     # week -> Current week title, eg:-  2016-06-10
#
#             bars = plot_bar_animated_altair(weekly_df, week)
#             time.sleep(0.01)
#
#             bar_plot.altair_chart(bars)
#
#             st.balloons()  # Displays some celebratory balloons for glamour!


print('To run: streamlit run streamlit_app.py')