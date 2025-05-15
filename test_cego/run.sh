#!/bin/bash

# This script is used to run the Python script with the specified arguments.
ARG=(
    "datasets/data_42v_20x20"
    "cfg_1"
)

python3 main.py "${ARG[@]}"