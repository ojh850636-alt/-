#!/usr/bin/env python3
from __future__ import annotations

import c58_behavior_causal as core
from c58_control_mutations import snapshot_adapter, randomize_inplace, dose_inplace

_SNAP={}

def _snap(model, adapter_name):
    key=(id(model),adapter_name)
    if key not in _SNAP:
        _SNAP[key]=snapshot_adapter(model,adapter_name)
    return _SNAP[key]


def _randomize(model,src_name,new_name,seed,mode):
    # `new_name` is intentionally ignored. The presealed transformation is applied
    # to the RTE adapter in place, scored immediately, then restored from the same
    # original snapshot before the next condition.
    randomize_inplace(model,src_name,_snap(model,src_name),seed,mode)


def _dose(model,src_name,new_name,scale):
    dose_inplace(model,src_name,_snap(model,src_name),scale)

core.randomize=_randomize
core.dose=_dose

if __name__=='__main__':
    core.main()
