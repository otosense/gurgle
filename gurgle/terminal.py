from lined.tools import generator_version
from lined import Pipeline
from functools import partial

import numpy as np


class DFLT:
    gain = 10
    cutoff = 50
    dispstr = '*'
    prefix = '|'
    fv_num_sep = '\n'


def viz_norm(x, gain=10, cutoff=50):
    return min(cutoff, x * gain)


def ascii_levels(x, viz_norm=viz_norm, dispstr=DFLT.dispstr, prefix=DFLT.prefix):
    return prefix + dispstr * int(viz_norm(x))


dflt_num_to_viz_str = ascii_levels


def multipart_fv_str(fv,
                     num_to_viz_str=dflt_num_to_viz_str,
                     fv_num_sep=DFLT.fv_num_sep):
    return fv_num_sep.join(map(dflt_num_to_viz_str, dflt_chk_to_fv(fv))) + '\n'


ascii_levels_fv_to_str = partial(multipart_fv_str,
                                 num_to_viz_str=ascii_levels,
                                 fv_num_sep=DFLT.fv_num_sep)

raw_fv_to_str = partial(multipart_fv_str,
                        num_to_viz_str=str,
                        fv_num_sep=DFLT.fv_num_sep)

dflt_fv_to_str = ascii_levels_fv_to_str

int16_max = np.iinfo(np.int16).max


def log_max_chk_to_fv(chk: np.ndarray, gain=50):
    """A chk_to_fv that has to do with volume"""
    return [np.log(max(chk.max(), abs(chk.min()), 1)) / np.log(int16_max) * gain]
    # return np.log(min(1, max(np.abs(chk)))) / np.log(int16_max) * gain


dflt_chk_to_fv = log_max_chk_to_fv


def chk_to_fv_viz(chk: np.ndarray,
                  chk_to_fv=dflt_chk_to_fv,
                  fv_to_str=dflt_fv_to_str):
    """Takes a chk and returns a string representing the fv. Meant to be displayed dynamically"""
    fv = chk_to_fv(chk)
    fv_str = fv_to_str(fv)
    return fv_str


from IPython.core.display import clear_output


def refresh_and_print(iterator_output):
    clear_output(wait=True)
    n, viz = iterator_output
    print(n, viz, sep='\n')


dflt_output_callback = refresh_and_print


def make_and_consume_iterator(iterator_func, *interator_func_args, **interator_func_kwargs):
    """make an consume and iterator"""
    for _ in iterator_func(*interator_func_args, **interator_func_kwargs):
        pass


# import time
#
# for x in range(0, 5):
#     b = "Loading" + "." * x
#     print(b, end="\r")
#     time.sleep(1)

from taped import LiveWf, WfChunks

from itertools import islice


def chk_std(chk):
    return np.std(chk)


same_line_print = partial(print, end='\r')

from typing import Optional


def launch(chk_to_fv=chk_std,
           fv_to_viz=print,
           input_device_index: Optional[int] = None,
           sr: int = 44100,
           chk_size: int = 2048,
           **kwargs
           ):
    """

    :param chk_to_fv:
    :param fv_to_viz:
    :return:
    """
    print("Starting...")
    with WfChunks(input_device_index=input_device_index,
                  sr=sr,
                  chk_size=chk_size,
                  **kwargs) as wf_chunks:
        try:
            for chk in wf_chunks:
                fv = chk_to_fv(chk)
                if fv is not None:
                    fv_to_viz(fv)

        except KeyboardInterrupt:
            print('KeyboardInterrupt: Okay... closing down...')

    print("... closing shop.")


same_line_print = partial(print, end='\r')


def log_print(fv):
    print(np.log2(1 + fv))


def exp_print(fv, e=2):
    print(fv ** e)


#### Gurgle ###############################
from functools import partial
from slang.spectrop import mk_chk_fft
from slang.chunkers import simple_fixed_step_chunker
from lined import Pipeline, iterize

chk_size = 2048
wf_to_chks = partial(simple_fixed_step_chunker, chk_size=chk_size)
chk_to_spectr = mk_chk_fft(chk_size)

from slang.spectrop import logarithmic_bands_matrix

M = logarithmic_bands_matrix(7)
bands_sp = partial(np.dot, b=M.T)

wf_to_spectr = Pipeline(wf_to_chks,
                        iterize(chk_to_spectr),
                        iterize(bands_sp))

from gurgle.sp_gurgle import PartialFitPipeline, ResidueGurgle, IncrementalPCA, ScaledIncrementalPCA
from lined import Pipeline
import operator

gurgle = ResidueGurgle(projector=ScaledIncrementalPCA(n_components=16))


def compute_outlier(chk):
    # fv = wf_to_spectr(chk)
    spectr = chk_to_spectr(chk)
    # return gurgle(spectr)
    fv, r = gurgle(spectr)
    if r is not None:
        return max(0, 2 + r[0]) ** 2
    else:
        return None


def print_ascii_levels_2_70(fv):
    print(ascii_levels(fv, viz_norm=partial(viz_norm, gain=2, cutoff=70)))


def print_ascii_levels_01_70(fv):
    print(ascii_levels(fv, viz_norm=partial(viz_norm, gain=0.1, cutoff=70)))


_allowed_objs = [
    chk_std,
    print,
    compute_outlier,
    print_ascii_levels_2_70,
    print_ascii_levels_01_70,
]


def allowed_objects():
    g = globals()
    for obj in _allowed_objs:
        if isinstance(obj, str) and obj in g:
            yield obj, g[obj]
        else:
            yield obj.__name__, obj


# def allowed_objects():
#     for k, v in globals().items():
#         if callable(v):
#             yield k, v

obj_store = dict(allowed_objects())


def main():
    import argh
    from gurgle.util import resolve_str_specification

    def list_possible_inputs():
        print(*obj_store, sep='\n')

    print("""
    Example usages:
    
    To get a list of elements:
        gurgle-terminal list-possible-inputs
    For volume tracking:
        gurgle-terminal launch chk_std print_ascii_levels_01_70
    For online spectral projector outliers tracking:
        gurgle-terminal launch compute_outlier print_ascii_levels_2_70

    """)

    argh.dispatch_commands([
        resolve_str_specification(obj_store)(launch),
        list_possible_inputs
    ])


if __name__ == '__main__':
    main()
