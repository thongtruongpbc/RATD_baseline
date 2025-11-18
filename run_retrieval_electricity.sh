#!/bin/bash
cd TCN-master
H=(96 192 336 720)
L=(720 96)
for l in "${L[@]}"; do
    for h in "${H[@]}"; do
        python3 retrieval.py --type encode  --datatype 'electricity' --L "$l" --H "$h"
        python3 retrieval.py --type retrieval  --datatype 'electricity' --L "$l" --H "$h"

    done
done
echo "All jobs started in background. Logs in logs/"
