cd /scratch1/yciftci/cs535project/myMMIM/src/slurm







# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/bert/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/bert/rnn/test_results/results.pickle" ]; then
# export log_file="mosi_none_bert.txt"
# export data="mosi"
# export text_encoder="bert"
# export fusion="none"
# sbatch --job-name mosi_none_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/none/bert"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/deberta/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/deberta/rnn/test_results/results.pickle" ]; then
# export log_file="mosi_none_deberta.txt"
# export data="mosi"
# export text_encoder="deberta"
# export fusion="none"
# sbatch --job-name mosi_none_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/none/deberta"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/roberta/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/roberta/rnn/test_results/results.pickle" ]; then
# export log_file="mosi_none_roberta.txt"
# export data="mosi"
# export text_encoder="roberta"
# export fusion="none"
# sbatch --job-name mosi_none_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/none/roberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/glove/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/glove/rnn/test_results/results.pickle" ]; then
# export log_file="mosi_none_glove.txt"
# export data="mosi"
# export text_encoder="glove"
# export fusion="none"
# sbatch --job-name mosi_none_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/none/glove"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/bert/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/bert/rnn/test_results/results.pickle" ]; then
# export log_file="mosei_none_bert.txt"
# export data="mosei"
# export text_encoder="bert"
# export fusion="none"
# sbatch --job-name mosei_none_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/none/bert"
# fi






# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/deberta/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/deberta/rnn/test_results/results.pickle" ]; then
# export log_file="mosei_none_deberta.txt"
# export data="mosei"
# export text_encoder="deberta"
# export audio_encoder="rnn"
# export fusion="none"
# sbatch --job-name mosei_none_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/none/deberta"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/roberta/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/roberta/rnn/test_results/results.pickle" ]; then
# export log_file="mosei_none_roberta.txt"
# export data="mosei"
# export text_encoder="roberta"
# export fusion="none"
# sbatch --job-name mosei_none_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/none/roberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/glove/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/glove/rnn/test_results/results.pickle" ]; then
# export log_file="mosei_none_glove.txt"
# export data="mosei"
# export text_encoder="glove"
# export fusion="none"
# sbatch --job-name mosei_none_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/none/glove"
# fi































# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/bert/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/bert/hubert/test_results/results.pickle" ]; then
# export log_file="mosi_none_bert_hubert.txt"
# export data="mosi"
# export audio_encoder="hubert"
# export text_encoder="bert"
# export fusion="none"
# sbatch --job-name mosi_none_bert_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/none/bert/hubert"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/deberta/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/deberta/hubert/test_results/results.pickle" ]; then
# export log_file="mosi_none_deberta_hubert.txt"
# export data="mosi"
# export audio_encoder="hubert"
# export text_encoder="deberta"
# export fusion="none"
# sbatch --job-name mosi_none_deberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/none/deberta/hubert"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/roberta/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/roberta/hubert/test_results/results.pickle" ]; then
# export log_file="mosi_none_roberta_hubert.txt"
# export data="mosi"
# export audio_encoder="hubert"
# export text_encoder="roberta"
# export fusion="none"
# sbatch --job-name mosi_none_roberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/none/roberta/hubert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/glove/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/none/glove/hubert/test_results/results.pickle" ]; then
# export log_file="mosi_none_glove_hubert.txt"
# export data="mosi"
# export audio_encoder="hubert"
# export text_encoder="glove"
# export fusion="none"
# sbatch --job-name mosi_none_glove_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/none/glove/hubert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/bert/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/bert/hubert/test_results/results.pickle" ]; then
# export log_file="mosei_none_bert_hubert.txt"
# export data="mosei"
# export audio_encoder="hubert"
# export text_encoder="bert"
# export fusion="none"
# sbatch --job-name mosei_none_bert_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/none/bert/hubert"
# fi







if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/deberta/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/deberta/hubert/test_results/results.pickle" ]; then
export log_file="mosei_none_deberta_hubert.txt"
export data="mosei"
export audio_encoder="hubert"
export text_encoder="deberta"
export fusion="none"
sbatch --job-name mosei_none_deberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
else
echo "Skipping due to existing mosei/none/deberta/hubert"
fi






# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/roberta/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/roberta/hubert/test_results/results.pickle" ]; then
# export log_file="mosei_none_roberta_hubert.txt"
# export data="mosei"
# export audio_encoder="hubert"
# export text_encoder="roberta"
# export fusion="none"
# sbatch --job-name mosei_none_roberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/none/roberta/hubert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/glove/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/none/glove/hubert/test_results/results.pickle" ]; then
# export log_file="mosei_none_glove_hubert.txt"
# export data="mosei"
# export audio_encoder="hubert"
# export text_encoder="glove"
# export fusion="none"
# sbatch --job-name mosei_none_glove_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/none/glove/hubert"
# fi































# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/bert/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/bert/rnn/test_results/results.pickle" ]; then
# export log_file="mosi_gb_bert.txt"
# export data="mosi"
# export text_encoder="bert"
# export fusion="gb"
# sbatch --job-name mosi_gb_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/gb/bert"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/deberta/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/deberta/rnn/test_results/results.pickle" ]; then
# export log_file="mosi_gb_deberta.txt"
# export data="mosi"
# export text_encoder="deberta"
# export fusion="gb"
# sbatch --job-name mosi_gb_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/gb/deberta"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/roberta/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/roberta/rnn/test_results/results.pickle" ]; then
# export log_file="mosi_gb_roberta.txt"
# export data="mosi"
# export text_encoder="roberta"
# export fusion="gb"
# sbatch --job-name mosi_gb_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/gb/roberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/glove/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/glove/rnn/test_results/results.pickle" ]; then
# export log_file="mosi_gb_glove.txt"
# export data="mosi"
# export text_encoder="glove"
# export fusion="gb"
# sbatch --job-name mosi_gb_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/gb/glove"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/bert/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/bert/rnn/test_results/results.pickle" ]; then
# export log_file="mosei_gb_bert.txt"
# export data="mosei"
# export text_encoder="bert"
# export fusion="gb"
# sbatch --job-name mosei_gb_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/gb/bert"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/deberta/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/deberta/rnn/test_results/results.pickle" ]; then
# export log_file="mosei_gb_deberta.txt"
# export data="mosei"
# export text_encoder="deberta"
# export fusion="gb"
# sbatch --job-name mosei_gb_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/gb/deberta"
# fi




# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/roberta/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/roberta/rnn/test_results/results.pickle" ]; then
# export log_file="mosei_gb_roberta.txt"
# export data="mosei"
# export text_encoder="roberta"
# export fusion="gb"
# sbatch --job-name mosei_gb_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/gb/roberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/glove/rnn/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/glove/rnn/test_results/results.pickle" ]; then
# export log_file="mosei_gb_glove.txt"
# export data="mosei"
# export text_encoder="glove"
# export fusion="gb"
# sbatch --job-name mosei_gb_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/gb/glove"
# fi



























# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/bert/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/bert/hubert/test_results/results.pickle" ]; then
# export log_file="mosi_gb_bert_hubert.txt"
# export data="mosi"
# export audio_encoder="hubert"
# export text_encoder="bert"
# export fusion="gb"
# sbatch --job-name mosi_gb_bert_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/gb/bert/hubert"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/deberta/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/deberta/hubert/test_results/results.pickle" ]; then
# export log_file="mosi_gb_deberta_hubert.txt"
# export data="mosi"
# export audio_encoder="hubert"
# export text_encoder="deberta"
# export fusion="gb"
# sbatch --job-name mosi_gb_deberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/gb/deberta/hubert"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/roberta/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/roberta/hubert/test_results/results.pickle" ]; then
# export log_file="mosi_gb_roberta_hubert.txt"
# export data="mosi"
# export audio_encoder="hubert"
# export text_encoder="roberta"
# export fusion="gb"
# sbatch --job-name mosi_gb_roberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/gb/roberta/hubert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/glove/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/gb/glove/hubert/test_results/results.pickle" ]; then
# export log_file="mosi_gb_glove_hubert.txt"
# export data="mosi"
# export audio_encoder="hubert"
# export text_encoder="glove"
# export fusion="gb"
# sbatch --job-name mosi_gb_glove_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/gb/glove/hubert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/bert/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/bert/hubert/test_results/results.pickle" ]; then
# export log_file="mosei_gb_bert_hubert.txt"
# export data="mosei"
# export audio_encoder="hubert"
# export text_encoder="bert"
# export fusion="gb"
# sbatch --job-name mosei_gb_bert_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/gb/bert/hubert"
# fi





# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/deberta/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/deberta/hubert/test_results/results.pickle" ]; then
# export log_file="mosei_gb_deberta_hubert.txt"
# export data="mosei"
# export audio_encoder="hubert"
# export text_encoder="deberta"
# export fusion="gb"
# sbatch --job-name mosei_gb_deberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/gb/deberta/hubert"
# fi




# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/roberta/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/roberta/hubert/test_results/results.pickle" ]; then
# export log_file="mosei_gb_roberta_hubert.txt"
# export data="mosei"
# export audio_encoder="hubert"
# export text_encoder="roberta"
# export fusion="gb"
# sbatch --job-name mosei_gb_roberta_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/gb/roberta/hubert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/glove/hubert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/gb/glove/hubert/test_results/results.pickle" ]; then
# export log_file="mosei_gb_glove_hubert.txt"
# export data="mosei"
# export audio_encoder="hubert"
# export text_encoder="glove"
# export fusion="gb"
# sbatch --job-name mosei_gb_glove_hubert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/gb/glove/hubert"
# fi
















































































# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/text/bert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/text/bert/test_results/results.pickle" ]; then
# export log_file="mosi_text_bert.txt"
# export data="mosi"
# export text_encoder="bert"
# export fusion="text"
# sbatch --job-name mosi_text_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/text/bert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/text/deberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/text/deberta/test_results/results.pickle" ]; then
# export log_file="mosi_text_deberta.txt"
# export data="mosi"
# export text_encoder="deberta"
# export fusion="text"
# sbatch --job-name mosi_text_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/text/deberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/text/roberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/text/roberta/test_results/results.pickle" ]; then
# export log_file="mosi_text_roberta.txt"
# export data="mosi"
# export text_encoder="roberta"
# export fusion="text"
# sbatch --job-name mosi_text_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/text/roberta"
# fi

# # if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/text/glove/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/text/glove/test_results/results.pickle" ]; then
# # export log_file="mosi_text_glove.txt"
# # export data="mosi"
# # export text_encoder="glove"
# # export fusion="text"
# # sbatch --job-name mosi_text_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# # else
# # echo "Skipping due to existing mosi/text/glove"
# # fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/text/bert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/text/bert/test_results/results.pickle" ]; then
# export log_file="mosei_text_bert.txt"
# export data="mosei"
# export text_encoder="bert"
# export fusion="text"
# sbatch --job-name mosei_text_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/text/bert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/text/deberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/text/deberta/test_results/results.pickle" ]; then
# export log_file="mosei_text_deberta.txt"
# export data="mosei"
# export text_encoder="deberta"
# export fusion="text"
# sbatch --job-name mosei_text_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/text/deberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/text/roberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/text/roberta/test_results/results.pickle" ]; then
# export log_file="mosei_text_roberta.txt"
# export data="mosei"
# export text_encoder="roberta"
# export fusion="text"
# sbatch --job-name mosei_text_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/text/roberta"
# fi

# # if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/text/glove/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/text/glove/test_results/results.pickle" ]; then
# # export log_file="mosei_text_glove.txt"
# # export data="mosei"
# # export text_encoder="glove"
# # export fusion="text"
# # sbatch --job-name mosei_text_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# # else
# # echo "Skipping due to existing mosei/text/glove"
# # fi
















# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/early/bert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/early/bert/test_results/results.pickle" ]; then
# export log_file="mosi_early_bert.txt"
# export data="mosi"
# export text_encoder="bert"
# export fusion="early"
# sbatch --job-name mosi_early_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/early/bert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/early/deberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/early/deberta/test_results/results.pickle" ]; then
# export log_file="mosi_early_deberta.txt"
# export data="mosi"
# export text_encoder="deberta"
# export fusion="early"
# sbatch --job-name mosi_early_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/early/deberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/early/roberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/early/roberta/test_results/results.pickle" ]; then
# export log_file="mosi_early_roberta.txt"
# export data="mosi"
# export text_encoder="roberta"
# export fusion="early"
# sbatch --job-name mosi_early_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/early/roberta"
# fi

# # if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/early/glove/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/early/glove/test_results/results.pickle" ]; then
# # export log_file="mosi_early_glove.txt"
# # export data="mosi"
# # export text_encoder="glove"
# # export fusion="early"
# # sbatch --job-name mosi_early_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# # else
# # echo "Skipping due to existing mosi/early/glove"
# # fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/early/bert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/early/bert/test_results/results.pickle" ]; then
# export log_file="mosei_early_bert.txt"
# export data="mosei"
# export text_encoder="bert"
# export fusion="early"
# sbatch --job-name mosei_early_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/early/bert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/early/deberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/early/deberta/test_results/results.pickle" ]; then
# export log_file="mosei_early_deberta.txt"
# export data="mosei"
# export text_encoder="deberta"
# export fusion="early"
# sbatch --job-name mosei_early_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/early/deberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/early/roberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/early/roberta/test_results/results.pickle" ]; then
# export log_file="mosei_early_roberta.txt"
# export data="mosei"
# export text_encoder="roberta"
# export fusion="early"
# sbatch --job-name mosei_early_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/early/roberta"
# fi

# # if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/early/glove/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/early/glove/test_results/results.pickle" ]; then
# # export log_file="mosei_early_glove.txt"
# # export data="mosei"
# # export text_encoder="glove"
# # export fusion="early"
# # sbatch --job-name mosei_early_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# # else
# # echo "Skipping due to existing mosei/early/glove"
# # fi
















# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/late/bert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/late/bert/test_results/results.pickle" ]; then
# export log_file="mosi_late_bert.txt"
# export data="mosi"
# export text_encoder="bert"
# export fusion="late"
# sbatch --job-name mosi_late_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/late/bert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/late/deberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/late/deberta/test_results/results.pickle" ]; then
# export log_file="mosi_late_deberta.txt"
# export data="mosi"
# export text_encoder="deberta"
# export fusion="late"
# sbatch --job-name mosi_late_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/late/deberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/late/roberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/late/roberta/test_results/results.pickle" ]; then
# export log_file="mosi_late_roberta.txt"
# export data="mosi"
# export text_encoder="roberta"
# export fusion="late"
# sbatch --job-name mosi_late_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/late/roberta"
# fi

# # if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/late/glove/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/late/glove/test_results/results.pickle" ]; then
# # export log_file="mosi_late_glove.txt"
# # export data="mosi"
# # export text_encoder="glove"
# # export fusion="late"
# # sbatch --job-name mosi_late_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# # else
# # echo "Skipping due to existing mosi/late/glove"
# # fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/late/bert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/late/bert/test_results/results.pickle" ]; then
# export log_file="mosei_late_bert.txt"
# export data="mosei"
# export text_encoder="bert"
# export fusion="late"
# sbatch --job-name mosei_late_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/late/bert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/late/deberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/late/deberta/test_results/results.pickle" ]; then
# export log_file="mosei_late_deberta.txt"
# export data="mosei"
# export text_encoder="deberta"
# export fusion="late"
# sbatch --job-name mosei_late_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/late/deberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/late/roberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/late/roberta/test_results/results.pickle" ]; then
# export log_file="mosei_late_roberta.txt"
# export data="mosei"
# export text_encoder="roberta"
# export fusion="late"
# sbatch --job-name mosei_late_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/late/roberta"
# fi

# # if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/late/glove/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/late/glove/test_results/results.pickle" ]; then
# # export log_file="mosei_late_glove.txt"
# # export data="mosei"
# # export text_encoder="glove"
# # export fusion="late"
# # sbatch --job-name mosei_late_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# # else
# # echo "Skipping due to existing mosei/late/glove"
# # fi














# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/audio/bert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/audio/bert/test_results/results.pickle" ]; then
# export log_file="mosi_audio_bert.txt"
# export data="mosi"
# export text_encoder="bert"
# export fusion="audio"
# sbatch --job-name mosi_audio_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/audio/bert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/audio/deberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/audio/deberta/test_results/results.pickle" ]; then
# export log_file="mosi_audio_deberta.txt"
# export data="mosi"
# export text_encoder="deberta"
# export fusion="audio"
# sbatch --job-name mosi_audio_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/audio/deberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/audio/roberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/audio/roberta/test_results/results.pickle" ]; then
# export log_file="mosi_audio_roberta.txt"
# export data="mosi"
# export text_encoder="roberta"
# export fusion="audio"
# sbatch --job-name mosi_audio_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/audio/roberta"
# fi

# # if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/audio/glove/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/audio/glove/test_results/results.pickle" ]; then
# # export log_file="mosi_audio_glove.txt"
# # export data="mosi"
# # export text_encoder="glove"
# # export fusion="audio"
# # sbatch --job-name mosi_audio_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# # else
# # echo "Skipping due to existing mosi/audio/glove"
# # fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/audio/bert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/audio/bert/test_results/results.pickle" ]; then
# export log_file="mosei_audio_bert.txt"
# export data="mosei"
# export text_encoder="bert"
# export fusion="audio"
# sbatch --job-name mosei_audio_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/audio/bert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/audio/deberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/audio/deberta/test_results/results.pickle" ]; then
# export log_file="mosei_audio_deberta.txt"
# export data="mosei"
# export text_encoder="deberta"
# export fusion="audio"
# sbatch --job-name mosei_audio_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/audio/deberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/audio/roberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/audio/roberta/test_results/results.pickle" ]; then
# export log_file="mosei_audio_roberta.txt"
# export data="mosei"
# export text_encoder="roberta"
# export fusion="audio"
# sbatch --job-name mosei_audio_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/audio/roberta"
# fi

# # if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/audio/glove/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/audio/glove/test_results/results.pickle" ]; then
# # export log_file="mosei_audio_glove.txt"
# # export data="mosei"
# # export text_encoder="glove"
# # export fusion="audio"
# # sbatch --job-name mosei_audio_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# # else
# # echo "Skipping due to existing mosei/audio/glove"
# # fi














# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/video/bert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/video/bert/test_results/results.pickle" ]; then
# export log_file="mosi_video_bert.txt"
# export data="mosi"
# export text_encoder="bert"
# export fusion="video"
# sbatch --job-name mosi_video_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/video/bert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/video/deberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/video/deberta/test_results/results.pickle" ]; then
# export log_file="mosi_video_deberta.txt"
# export data="mosi"
# export text_encoder="deberta"
# export fusion="video"
# sbatch --job-name mosi_video_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/video/deberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/video/roberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/video/roberta/test_results/results.pickle" ]; then
# export log_file="mosi_video_roberta.txt"
# export data="mosi"
# export text_encoder="roberta"
# export fusion="video"
# sbatch --job-name mosi_video_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosi/video/roberta"
# fi

# # if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosi/video/glove/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosi/video/glove/test_results/results.pickle" ]; then
# # export log_file="mosi_video_glove.txt"
# # export data="mosi"
# # export text_encoder="glove"
# # export fusion="video"
# # sbatch --job-name mosi_video_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# # else
# # echo "Skipping due to existing mosi/video/glove"
# # fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/video/bert/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/video/bert/test_results/results.pickle" ]; then
# export log_file="mosei_video_bert.txt"
# export data="mosei"
# export text_encoder="bert"
# export fusion="video"
# sbatch --job-name mosei_video_bert --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/video/bert"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/video/deberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/video/deberta/test_results/results.pickle" ]; then
# export log_file="mosei_video_deberta.txt"
# export data="mosei"
# export text_encoder="deberta"
# export fusion="video"
# sbatch --job-name mosei_video_deberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/video/deberta"
# fi

# if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/video/roberta/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/video/roberta/test_results/results.pickle" ]; then
# export log_file="mosei_video_roberta.txt"
# export data="mosei"
# export text_encoder="roberta"
# export fusion="video"
# sbatch --job-name mosei_video_roberta --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# else
# echo "Skipping due to existing mosei/video/roberta"
# fi

# # if [ ! -f "/scratch1/yciftci/cs535project/myMMIM/r/mosei/video/glove/test_results/results.pickle" ] || [ ! -s "/scratch1/yciftci/cs535project/myMMIM/r/mosei/video/glove/test_results/results.pickle" ]; then
# # export log_file="mosei_video_glove.txt"
# # export data="mosei"
# # export text_encoder="glove"
# # export fusion="video"
# # sbatch --job-name mosei_video_glove --time 48:00:00 --partition gpu --cpus-per-task 1 --mem 32GB --gres=gpu:a40:1 --mail-user yciftci@usc.edu mmim.slurm;
# # else
# # echo "Skipping due to existing mosei/video/glove"
# # fi










