#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
python  main.py --cfg ./config/train.yaml
