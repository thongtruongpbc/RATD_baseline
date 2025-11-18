#!/bin/bash

H=(96 192 336 720)
L=(96 720)
for h in "${H[@]}"; do
    for l in "${L[@]}"; do
        python exe_forecasting.py --datatype ETTh1  --h_size "$l" --ref_size "$h"

    done
done
echo "All jobs started in background. Logs in logs/"
