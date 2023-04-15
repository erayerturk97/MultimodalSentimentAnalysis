#!/bin/bash 

# inputs to slurm (should be exported as enviroment variables)
: "${Q:=shanechi}"
: "${TIME:=72:00:00}"
: "${MEM:=32GB}"

: "${env_dir:=/scratch1/yciftci/cs535project/mmim_venv/bin/activate}"
: "${log_dir:=/scratch1/yciftci/cs535project/myMMIM/logs}"
: "${log_file:=log.txt}"
: "${code_dir:=/scratch1/yciftci/cs535project/myMMIM/src}"
: "${text_encoder:=bert}"
: "${data:=mosi}"
: "${fusion:=none}"

# For discovery
module purge
module load gcc/8.3.0
module load cuda/10.0.130
module load cudnn/7.5.1-10.1
module load python/3.7.6

mkdir -p $log_dir
echo $env_dir
source $env_dir
export MPLCONFIGDIR=/scratch1/yciftci/python-cache/mpl-cache
export GIT_PYTHON_REFRESH=quiet

cd $code_dir
python3 -u main.py --text_encoder ${text_encoder} --dataset ${data} --fusion ${fusion}| tee -a ${log_dir}/${log_file}