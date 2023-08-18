#!/bin/bash


for seed in 0 1 2
do 
    for task in Goal # Circle Push
    do 
        for robot in Point Car # Doggo
        do
            for level in 1 2
            do 
                for algo in ppo sac dreamer_v3 p2e_dv2
                do
                    env_id="Wrapped-Safety${robot}${task}${level}-v0"
                    lightning run model --accelerator=gpu --strategy=ddp --devices=1 \
                        svf_gymnasium/sheeprl/train.py ${algo} \
                        --env $env_id \
                        --seed $seed \
                        --track \
                        --wandb_group safety-violation-expt \
                        --total_step 1000000
                done
            done
        done
    done
done