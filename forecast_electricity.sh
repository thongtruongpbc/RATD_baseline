#!/bin/bash

H=(96 192 336 720)
L=(720 96)
for l in "${L[@]}"; do
    for h in "${H[@]}"; do
        python exe_forecasting.py --datatype electricity  --h_size "$l" --ref_size "$h"

    done
done
echo "All jobs started in background. Logs in logs/"
