#!/bin/bash
folds='1 2 3 4 5'
for fold in $folds
do
    python3 test.py $fold
done