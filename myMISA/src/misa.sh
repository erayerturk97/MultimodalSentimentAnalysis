export log_file="mosi_bert.txt"
export data="mosi"
export text_encoder="bert"
export audio_encoder="rnn"
sbatch --job-name mosi_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosi_deberta.txt"
export data="mosi"
export text_encoder="deberta"
export audio_encoder="rnn"
sbatch --job-name mosi_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosi_roberta.txt"
export data="mosi"
export text_encoder="roberta"
export audio_encoder="rnn"
sbatch --job-name mosi_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosi_glove.txt"
export data="mosi"
export text_encoder="glove"
export audio_encoder="rnn"
sbatch --job-name mosi_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosei_bert.txt"
export data="mosei"
export text_encoder="bert"
export audio_encoder="rnn"
sbatch --job-name mosei_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosei_deberta.txt"
export data="mosei"
export text_encoder="deberta"
export audio_encoder="rnn"
sbatch --job-name mosei_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosei_roberta.txt"
export data="mosei"
export text_encoder="roberta"
export audio_encoder="rnn"
sbatch --job-name mosei_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosei_glove.txt"
export data="mosei"
export text_encoder="glove"
export audio_encoder="rnn"
sbatch --job-name mosei_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;


export log_file="mosi_bert_hubert.txt"
export data="mosi"
export text_encoder="bert"
export audio_encoder="hubert"
sbatch --job-name mosi_bert_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosi_deberta_hubert.txt"
export data="mosi"
export text_encoder="deberta"
export audio_encoder="hubert"
sbatch --job-name mosi_deberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosi_roberta_hubert.txt"
export data="mosi"
export text_encoder="roberta"
export audio_encoder="hubert"
sbatch --job-name mosi_roberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosi_glove_hubert.txt"
export data="mosi"
export text_encoder="glove"
export audio_encoder="hubert"
sbatch --job-name mosi_glove_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosei_bert_hubert.txt"
export data="mosei"
export text_encoder="bert"
export audio_encoder="hubert"
sbatch --job-name mosei_bert_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosei_deberta_hubert.txt"
export data="mosei"
export text_encoder="deberta"
export audio_encoder="hubert"
sbatch --job-name mosei_deberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosei_roberta_hubert.txt"
export data="mosei"
export text_encoder="roberta"
export audio_encoder="hubert"
sbatch --job-name mosei_roberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

export log_file="mosei_glove_hubert.txt"
export data="mosei"
export text_encoder="glove"
export audio_encoder="hubert"
sbatch --job-name mosei_glove_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu misa.slurm;

