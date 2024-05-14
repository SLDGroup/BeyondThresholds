#!/bin/bash

datasets=("dsads" "opportunity" "rwhar")
seeds=(123 456 789)
policies=("random" "threshold" "optimal_U" "optimal_1" "optimal_5" "optimal")

# derive policies
for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        for policy in "${policies[@]}"; do
            python derive_policies.py --dataset "$dataset" --seed "$seed" --policy "$policy"
        done
    done
done

# evaluating policies
for dataset in "${datasets[@]}"; do
    for seed in "${seeds[@]}"; do
        for policy in "${policies[@]}"; do
            python evaluate_policies.py --dataset "$dataset" --seed "$seed" --policy "$policy"
        done
    done
done