# !/bin/bash

# Times: Year Month Day Hour Minute Second
ts=`date +%Y%m%d%H%m%s`

# main
nohup python train.py > ./logs/${ts}.log 2>&1 &
