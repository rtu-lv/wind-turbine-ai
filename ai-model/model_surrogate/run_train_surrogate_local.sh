#!/bin/bash
python3 model_train_surrogate.py --model cnnA --data surrDS.bin --epochs 5000 --trials 10 --db_path /path_to_alya_data/