#!/bin/bash

gpu=0

for domain in w2a w2d d2w d2a a2w a2d
do
    #python pfc_source.py --dset $domain --gpu_id $gpu --office31
    python pfc_target.py --office31 --dset $domain --gpu_id $gpu
done
