# text
export log_file="mosi_bert_text.txt"
export data="mosi"
export text_encoder="bert"
export audio_encoder="rnn"
export model="TextClassifier"
sbatch --job-name mosi_bert_text --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_deberta_text.txt"
export data="mosi"
export text_encoder="deberta"
export audio_encoder="rnn"
export model="TextClassifier"
sbatch --job-name mosi_deberta_text --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_roberta_text.txt"
export data="mosi"
export text_encoder="roberta"
export audio_encoder="rnn"
export model="TextClassifier"
sbatch --job-name mosi_roberta_text --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_glove_text.txt"
export data="mosi"
export text_encoder="glove"
export audio_encoder="rnn"
export model="TextClassifier"
sbatch --job-name mosi_glove_text --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_bert_text.txt"
export data="mosei"
export text_encoder="bert"
export audio_encoder="rnn"
export model="TextClassifier"
sbatch --job-name mosei_bert_text --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_deberta_text.txt"
export data="mosei"
export text_encoder="deberta"
export audio_encoder="rnn"
export model="TextClassifier"
sbatch --job-name mosei_deberta_text --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_roberta_text.txt"
export data="mosei"
export text_encoder="roberta"
export audio_encoder="rnn"
export model="TextClassifier"
sbatch --job-name mosei_roberta_text --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_glove_text.txt"
export data="mosei"
export text_encoder="glove"
export audio_encoder="rnn"
export model="TextClassifier"
sbatch --job-name mosei_glove_text --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

# early fusion - rnn audio
export log_file="mosi_bert_early.txt"
export data="mosi"
export text_encoder="bert"
export audio_encoder="rnn"
export model="EarlyFusion"
sbatch --job-name mosi_bert_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_deberta_early.txt"
export data="mosi"
export text_encoder="deberta"
export audio_encoder="rnn"
export model="EarlyFusion"
sbatch --job-name mosi_deberta_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_roberta_early.txt"
export data="mosi"
export text_encoder="roberta"
export audio_encoder="rnn"
export model="EarlyFusion"
sbatch --job-name mosi_roberta_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_glove_early.txt"
export data="mosi"
export text_encoder="glove"
export audio_encoder="rnn"
export model="EarlyFusion"
sbatch --job-name mosi_glove_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_bert_early.txt"
export data="mosei"
export text_encoder="bert"
export audio_encoder="rnn"
export model="EarlyFusion"
sbatch --job-name mosei_bert_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_deberta_early.txt"
export data="mosei"
export text_encoder="deberta"
export audio_encoder="rnn"
export model="EarlyFusion"
sbatch --job-name mosei_deberta_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_roberta_early.txt"
export data="mosei"
export text_encoder="roberta"
export audio_encoder="rnn"
export model="EarlyFusion"
sbatch --job-name mosei_roberta_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_glove_early.txt"
export data="mosei"
export text_encoder="glove"
export audio_encoder="rnn"
export model="EarlyFusion"
sbatch --job-name mosei_glove_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

# early fusion - hubert audio
export log_file="mosi_bert_hubert_early.txt"
export data="mosi"
export text_encoder="bert"
export audio_encoder="hubert"
export model="EarlyFusion"
sbatch --job-name mosi_bert_hubert_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_deberta_hubert_early.txt"
export data="mosi"
export text_encoder="deberta"
export audio_encoder="hubert"
export model="EarlyFusion"
sbatch --job-name mosi_deberta_hubert_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_roberta_hubert_early.txt"
export data="mosi"
export text_encoder="roberta"
export audio_encoder="hubert"
export model="EarlyFusion"
sbatch --job-name mosi_roberta_hubert_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_glove_hubert_early.txt"
export data="mosi"
export text_encoder="glove"
export audio_encoder="hubert"
export model="EarlyFusion"
sbatch --job-name mosi_glove_hubert_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_bert_hubert_early.txt"
export data="mosei"
export text_encoder="bert"
export audio_encoder="hubert"
export model="EarlyFusion"
sbatch --job-name mosei_bert_hubert_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_deberta_hubert_early.txt"
export data="mosei"
export text_encoder="deberta"
export audio_encoder="hubert"
export model="EarlyFusion"
sbatch --job-name mosei_deberta_hubert_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_roberta_hubert_early.txt"
export data="mosei"
export text_encoder="roberta"
export audio_encoder="hubert"
export model="EarlyFusion"
sbatch --job-name mosei_roberta_hubert_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_glove_hubert_early.txt"
export data="mosei"
export text_encoder="glove"
export audio_encoder="hubert"
export model="EarlyFusion"
sbatch --job-name mosei_glove_hubert_early --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;


# late fusion - rnn audio
export log_file="mosi_bert_late.txt"
export data="mosi"
export text_encoder="bert"
export audio_encoder="rnn"
export model="LateFusion"
sbatch --job-name mosi_bert_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_deberta_late.txt"
export data="mosi"
export text_encoder="deberta"
export audio_encoder="rnn"
export model="LateFusion"
sbatch --job-name mosi_deberta_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_roberta_late.txt"
export data="mosi"
export text_encoder="roberta"
export audio_encoder="rnn"
export model="LateFusion"
sbatch --job-name mosi_roberta_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_glove_late.txt"
export data="mosi"
export text_encoder="glove"
export audio_encoder="rnn"
export model="LateFusion"
sbatch --job-name mosi_glove_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_bert_late.txt"
export data="mosei"
export text_encoder="bert"
export audio_encoder="rnn"
export model="LateFusion"
sbatch --job-name mosei_bert_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_deberta_late.txt"
export data="mosei"
export text_encoder="deberta"
export audio_encoder="rnn"
export model="LateFusion"
sbatch --job-name mosei_deberta_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_roberta_late.txt"
export data="mosei"
export text_encoder="roberta"
export audio_encoder="rnn"
export model="LateFusion"
sbatch --job-name mosei_roberta_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_glove_late.txt"
export data="mosei"
export text_encoder="glove"
export audio_encoder="rnn"
export model="LateFusion"
sbatch --job-name mosei_glove_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

# late fusion - hubert audio
export log_file="mosi_bert_hubert_late.txt"
export data="mosi"
export text_encoder="bert"
export audio_encoder="hubert"
export model="LateFusion"
sbatch --job-name mosi_bert_hubert_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_deberta_hubert_late.txt"
export data="mosi"
export text_encoder="deberta"
export audio_encoder="hubert"
export model="LateFusion"
sbatch --job-name mosi_deberta_hubert_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_roberta_hubert_late.txt"
export data="mosi"
export text_encoder="roberta"
export audio_encoder="hubert"
export model="LateFusion"
sbatch --job-name mosi_roberta_hubert_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosi_glove_hubert_late.txt"
export data="mosi"
export text_encoder="glove"
export audio_encoder="hubert"
export model="LateFusion"
sbatch --job-name mosi_glove_hubert_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_bert_hubert_late.txt"
export data="mosei"
export text_encoder="bert"
export audio_encoder="hubert"
export model="LateFusion"
sbatch --job-name mosei_bert_hubert_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_deberta_hubert_late.txt"
export data="mosei"
export text_encoder="deberta"
export audio_encoder="hubert"
export model="LateFusion"
sbatch --job-name mosei_deberta_hubert_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_roberta_hubert_late.txt"
export data="mosei"
export text_encoder="roberta"
export audio_encoder="hubert"
export model="LateFusion"
sbatch --job-name mosei_roberta_hubert_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_glove_hubert_late.txt"
export data="mosei"
export text_encoder="glove"
export audio_encoder="hubert"
export model="LateFusion"
sbatch --job-name mosei_glove_hubert_late --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

# rnn acoustic
export log_file="mosi_acoustic.txt"
export data="mosi"
export text_encoder="bert"
export audio_encoder="rnn"
export model="AcousticClassifier"
sbatch --job-name mosi_acoustic --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_acoustic.txt"
export data="mosei"
export text_encoder="bert"
export audio_encoder="rnn"
export model="AcousticClassifier"
sbatch --job-name mosei_acoustic --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

# hubert acoustic
export log_file="mosi_hubert_acoustic.txt"
export data="mosi"
export text_encoder="bert"
export audio_encoder="hubert"
export model="AcousticClassifier"
sbatch --job-name mosi_hubert_acoustic --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_hubert_acoustic.txt"
export data="mosei"
export text_encoder="bert"
export audio_encoder="hubert"
export model="AcousticClassifier"
sbatch --job-name mosei_hubert_acoustic --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

# visual
export log_file="mosi_visual.txt"
export data="mosi"
export text_encoder="bert"
export audio_encoder="rnn"
export model="VisualClassifier"
sbatch --job-name mosi_visual --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;

export log_file="mosei_visual.txt"
export data="mosei"
export text_encoder="bert"
export audio_encoder="rnn"
export model="VisualClassifier"
sbatch --job-name mosei_visual --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 36GB --gres=gpu:a100:1 --mail-user eerturk@usc.edu fusions.slurm;