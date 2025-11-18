#!/bin/bash

datatype=('weather' 'ETTh1' 'electricity')
H=(96 192 336 720)
L=(96 720)

for h in "${H[@]}"; do
    for l in "${L[@]}"; do
      for data in "${datatype[@]}"; do
        python exe_forecasting.py --datatype "$data"  --h_size "$l" --ref_size "$h" 
      done
    done
done
echo "All jobs started in background. Logs in logs/"
