#!/bin/bash

git clone https://github.com/facebookresearch/xformers.git --depth 1
cd xformers
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .
