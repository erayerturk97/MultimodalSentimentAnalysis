sbatch --job-name mosei_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu extract.slurm;
