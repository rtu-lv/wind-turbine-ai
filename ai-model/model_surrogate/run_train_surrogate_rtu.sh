#!/bin/sh
#python3 model_train_surrogate.py --model cnnA.pth --data surrDS.bin --epochs 10 --trials 10 \
#--num_cpus $PBS_NUM_PPN --num_gpus 1 --db_path /mnt/home/arnisl/raise/wind-turbine-ai/data

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate raise

cd ~/projects/raise/wind-turbine-ai/ai-model/model_surrogate
echo "Script executed from: ${PWD}"
 
python3 -u model_train_surrogate.py --model cnnA.pth --data surrDS2.bin --epochs 20 --trials 1 --db_path /home/raise/projects/raise/wind-turbine-ai/data/timeseries

echo "*** finished ****""
