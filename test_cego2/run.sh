#!/bin/bash

# This script is used to run the Python script with the specified arguments.
ARG=(
    "datasets/data_430v_100x100"
    "cfg_1"
)

python3 main.py "${ARG[@]}"