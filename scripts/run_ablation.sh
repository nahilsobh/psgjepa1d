#!/usr/bin/env bash
# Upstream Table-5 ablation, adapted to the 1-D car.
set -e
python train.py loss.grounding.weight=0.0                                            # LeWM-1D baseline
python train.py loss.grounding.use_transition=false loss.grounding.use_velocity=false # static only
python train.py loss.grounding.use_static=false                                       # transition only
python train.py                                                                       # full PSG-JEPA-1D
