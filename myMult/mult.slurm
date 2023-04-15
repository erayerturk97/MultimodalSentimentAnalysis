#!/bin/bash 

# inputs to slurm (should be exported as enviroment variables)
: "${Q:=shanechi}"
: "${TIME:=72:00:00}"
: "${MEM:=32GB}"

: "${env_dir:=/scratch1/eerturk/envs/venv_535_project/bin/activate}"
: "${log_dir:=/scratch2/eerturk/CSCI535/Project/myMult/logs}"
: "${log_file:=log.txt}"
: "${code_dir:=/scratch2/eerturk/CSCI535/Project/myMult}"
: "${text_encoder:=bert}"
: "${dataset:=mosi}"
: "${batch_size:=32}"
: "${num_epochs:=40}"
: "${lr:=1e-5}"

# For discovery
module purge
module load gcc/8.3.0
module load cuda/10.0.130
module load cudnn/7.5.1-10.1

mkdir -p $log_dir
echo $env_dir
source $env_dir 
export MPLCONFIGDIR=/scratch2/eerturk/python-cache/mpl-cache
export GIT_PYTHON_REFRESH=quiet

cd $code_dir
python3 -u main.py --text_encoder ${text_encoder} --dataset ${dataset} --batch_size ${batch_size} --lr ${lr} --num_epochs ${num_epochs}| tee -a ${log_dir}/${log_file}