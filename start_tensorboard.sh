#!/usr/bin/env bash
eval "$(conda shell.bash hook)"
conda activate apex
tensorboard --bind_all --logdir trained_models/ppo