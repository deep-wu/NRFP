#!/bin/bash

gpu=0
beta=1.8
alpha=1.8

for domain in a2c
do
    #python pfc_source.py --dset $domain --gpu_id $gpu --home
    python pfc_target_aug.py --home --dset $domain --gpu_id $gpu --sim_hyper $beta --iter_num=50 --dis_hyper $alpha --file time50 --aux_dataset_path '/24085404041/shot_Trans/data/office-home/aux_list.txt'
done
 