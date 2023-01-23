#!/bin/sh
#python3 model_train_surrogate.py --model cnnA --plot cnnA_2 --data surrDS.bin --epochs 5000
python3 model_train_surrogate.py --model cnnA.pth --data surrDS.bin --epochs 10 --trials 10 \
--num_cpus $PBS_NUM_PPN --num-gpus 1 --db_path /mnt/home/arnisl/raise/wind-turbine-ai/data