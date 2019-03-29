#!/bin/bash
if [ "$1" -eq 1 ]; then
    python a2_1.py $2 $3 $4
elif [ "$1" -eq 2 ]; then
    python a2_2.py $2 $3 $4 $5
fi
