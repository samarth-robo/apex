#!/usr/bin/env bash
conda activate apex
tensorboard --bind-all --logdir trained_models
