#!/bin/bash

# Script to reproduce results

for ((i=0;i<10;i+=1)); do
	python3 main.py \
		--save_models \
		--policy_name "TD3" \
		--env_name "Reacher-v2" \
		--seed $i \
		--start_timesteps 1000

	python3 main.py \
		--save_models \
		--policy_name "DDPG" \
		--env_name "Reacher-v2" \
		--seed $i \
		--start_timesteps 1000

	python3 main.py \
		--save_models \
		--policy_name "OurDDPG" \
		--env_name "Reacher-v2" \
		--seed $i \
		--start_timesteps 1000
done
