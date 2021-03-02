from typing import TypeVar, Optional, Iterable, Sequence, Callable
from dataclasses import dataclass

from slang.stypes import (
    Waveform,
    Waveforms,
    Chunker,
    Chunk,
    Chunks,
    Featurizer,
    Feature,
    FV,
    FVs,
    Quantizer,
    Snip,
    Snips,
    Snipper)

from slang.chunkers import fixed_step_chunker
from slang import (
    fixed_step_chunker,
    DFLT_CHUNKER,
    DFLT_CHK_SIZE,
    DFLT_WF_TO_SPECTR,
    SpectralProjector,
    DfltWfToChk,
    DfltChkToFv,
    KMeansFvToSnipDist,
)

from lined import Pipeline, iterize


@dataclass
class WfToFv:
    wf_to_chk: Chunker
    chk_to_fv: Featurizer

    def __call__(self, chk):
        return self.chk_to_fv(self.wf_to_chk(chk))


from slang.snip_stats import ClassifiedMomentsFitter
from functools import cached_property


# from slang.util import lazyprop as cached_property


@dataclass
class Gurgle:
    wf_to_chks: Chunker = DfltWfToChk()
    chk_to_fv: Featurizer = SpectralProjector()
    fv_to_snip: Quantizer = KMeansFvToSnipDist()
    snip_stats: ClassifiedMomentsFitter = ClassifiedMomentsFitter

    def __post_init__(self):
        # TODO: Handle all the "if type -- aligned with previous component"
        if isinstance(self.snip_stats, type):
            self.snip_stats = self.snip_stats(n_classes_=self.fv_to_snip.alphabet_size)

    def __call__(self, chk):
        fv = self.chk_to_fv(chk)
        snip, dist = self.fv_to_snip(fv)  # gfv stands for "grouped
        self.snip_stats.fit_partial_single((snip, dist))
        return self.snip_stats((snip, dist))

    @cached_property
    def wf_to_fvs(self):
        return Pipeline(self.wf_to_chks, iterize(self.chk_to_fv))
