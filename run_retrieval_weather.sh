#!/bin/bash
cd TCN-master
H=(96 192 336 720)
L=(96 720)
for h in "${H[@]}"; do
    for l in "${L[@]}"; do
        python3 retrieval.py --type encode  --datatype 'weather' --L "$l" --H "$h"
        python3 retrieval.py --type retrieval  --datatype 'weather' --L "$l" --H "$h"

    done
done
echo "All jobs started in background. Logs in logs/"
