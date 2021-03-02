
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


def streaming_graph(data_gen, window_length=5, inter_tick_sleep=0.01, max_y=10):
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

    i = 0
    for interval in sliding_window_intervals:
        line.set_ydata(interval)
        the_plot.pyplot(plt)
        time.sleep(inter_tick_sleep)
        i += 1


from taped import LiveWf, WfChunks


with WfChunks() as wf_chunks:
    def gen(x):
        for chk in islice(wf_chunks, 0, x):
            fv = np.std(chk)
            yield fv
    try:
        for fv in gen(100):
            print(fv)

    except KeyboardInterrupt:
        print('KeyboardInterrupt: Okay... closing down...')









