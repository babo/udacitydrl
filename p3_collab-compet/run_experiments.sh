#!/bin/bash

# Script to reproduce results

for ((i=0;i<10;i+=1)); do
	python3 main.py \
		--save_models \
		--policy_name "mDDPG" \
		--env_name "Tennis" \
		--seed $i \
		--start_timesteps 1000

	python3 main.py \
		--save_models \
		--policy_name "TD3" \
		--env_name "Tennis" \
		--seed $i \
		--start_timesteps 1000
done
