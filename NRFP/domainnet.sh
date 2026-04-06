#!/bin/bash

gpu=0
beta=2.6

for domain in p2c p2r p2s s2p s2r s2c r2s r2p r2c c2s c2p c2r c2s
do
    #python pfc_source.py --dset $domain --domainnet --gpu_id $gpu
    python /24085404041/PFC/PFC-main/pfc_target.py --dset $domain --gpu_id $gpu --domainnet --sim_hyper $beta --file origin_log
done
