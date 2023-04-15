# export log_file="mosi_bert.txt"
# export data="mosi"
# export text_encoder="bert"
# sbatch --job-name mosi_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

# export log_file="mosi_deberta.txt"
# export data="mosi"
# export text_encoder="deberta"
# sbatch --job-name mosi_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

# export log_file="mosi_roberta.txt"
# export data="mosi"
# export text_encoder="roberta"
# sbatch --job-name mosi_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

# export log_file="mosi_glove.txt"
# export data="mosi"
# export text_encoder="glove"
# sbatch --job-name mosi_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

# export log_file="mosei_bert.txt"
# export data="mosei"
# export text_encoder="bert"
# sbatch --job-name mosei_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosei_deberta.txt"
export data="mosei"
export text_encoder="deberta"
sbatch --job-name mosei_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

# export log_file="mosei_roberta.txt"
# export data="mosei"
# export text_encoder="roberta"
# sbatch --job-name mosei_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

# export log_file="mosei_glove.txt"
# export data="mosei"
# export text_encoder="glove"
# sbatch --job-name mosei_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

