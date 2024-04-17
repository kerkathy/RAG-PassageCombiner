set -e

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="3,5"

debug_mode=false

if [ "$#" -ne 3 ]; then
  echo "Illegal number of parameters"
  echo "Usage: $0 <dataset_short_name> <algo> <method>"
  exit 1
fi

# method should be either tfidf or sbert
if [ "$3" != "tfidf" ] && [ "$3" != "sbert" ]; then
  echo "Invalid method"
  exit 1
fi

dataset_short_name=$1
# "nq-test" "dpr-trivia-test" "hotpot" "msmarcoqa" "eli5" "strategyQA" 
algo=$2
# "basic" "mmr" "kmeans" "activeRDD"
method=$3
# "tfidf" "sbert"

# main_name="trivia-test" "my_orig_para1-nq-test"

num_docs=2
max_token=10

case $dataset_short_name in
  "msmarcoqa")
    num_docs=5
    max_token=100
    ;;
  "eli5")
    num_docs=10
    max_token=150
    ;;
  "strategyQA")
    num_docs=5
    max_token=150
    ;;
esac


# corpus_name=wiki
# corpus_name=web
# corpus_name=wiki-web
corpus_name=ms2
dataset_path=reproduce_retrieval/result/${dataset_short_name}/formatted-${corpus_name}.${dataset_short_name}.hits-100.json
output_dir="output_${corpus_name}_${dataset_short_name}_docs_${num_docs}"
lambda_param=0.5
threshold=1
# model="meta-llama/Llama-2-7b-hf"
model="huggyllama/llama-7b"
# huggyllama/llama-7b

print_and_run() {
    reranked_file=$1
    reranked_output_dir=$2

    # print all variable
    echo "----------------------------------------"
    printf "%-25s\t%s\n" "dataset_short_name:" "$dataset_short_name"
    printf "%-25s\t%s\n" "corpus_name:" "$corpus_name"
    printf "%-25s\t%s\n" "dataset_path:" "$reranked_file"
    printf "%-25s\t%s\n" "output_dir:" "$reranked_output_dir"
    printf "%-25s\t%s\n" "num_docs:" "$num_docs"
    printf "%-25s\t%s\n" "max_token:" "$max_token"
    echo "----------------------------------------"
    
    # --model_name \
    python eval_qa.py \
        --model_name $model \
        --dataset_path $reranked_file \
        --output_dir $reranked_output_dir \
        --num_docs $num_docs \
        --cache_dir cache \
        --max_token $max_token \
        --model_parallelism
}

start_time=$(date +%s)

if [ "$algo" = "mmr" ]; then
    for lambda_param in 0.5 0.6 0.7 0.8 0.9 1.0; do
        reranked_file=${dataset_path/.json/-reranked-mmr-${lambda_param}-${method}-${threshold}.json}
        # TODO 下面這段改更好
        reranked_output_dir=${output_dir}/reranked-mmr-${lambda_param}-${method}-${threshold}
        if [ $model == "meta-llama/Llama-2-7b-hf" ]; then
            reranked_output_dir=${output_dir}/reranked-mmr-${lambda_param}-${method}-${threshold}-llama2
        fi
        print_and_run $reranked_file $reranked_output_dir
    done
elif [ "$algo" = "basic" ]; then
    for threshold in 0.5 0.6 0.7 0.8 0.9 1; do
        reranked_file=${dataset_path/.json/-reranked-${method}-${threshold}.json}
        reranked_output_dir=${output_dir}/reranked_${method}-${threshold}
        if [ $threshold == 1 ]; then
            reranked_file=$dataset_path
            reranked_output_dir=${output_dir}/no_reranked
        fi
        # TODO: add llama2
        print_and_run $reranked_file $reranked_output_dir
    done
elif [ "$algo" = "kmeans" ]; then
    for k in 2 4; do
        reranked_file=${dataset_path/.json/-reranked-kmeans-${k}-${method}-${threshold}.json}
        reranked_output_dir=${output_dir}/reranked-kmeans-${k}-${method}-${threshold}
        if [ $model == "meta-llama/Llama-2-7b-hf" ]; then
            reranked_output_dir=${reranked_output_dir}-llama2
        fi
        if [ $debug_mode == true ]; then
            reranked_file=${reranked_file/.json/-debug.json}
            reranked_output_dir=${reranked_output_dir}-debug
        fi
        print_and_run $reranked_file $reranked_output_dir
    done
elif [ "$algo" = "activeRDD" ]; then
    for alpha in $(seq 0.0 0.1 1); do
        for beta in $(seq 0.0 0.1 1); do
            # skip if alpha + beta > 1
            if (( $(echo "$alpha + $beta > 1" | bc -l) )); then
                continue
            fi
            reranked_file=${dataset_path/.json/-reranked-activeRDD-${alpha}-${beta}-${method}-${threshold}.json}
            reranked_output_dir=${output_dir}/reranked-activeRDD-${alpha}-${beta}-${method}-${threshold}
            if [ $model == "meta-llama/Llama-2-7b-hf" ]; then
                reranked_output_dir=${reranked_output_dir}-llama2
            fi
            if [ $debug_mode == true ]; then
                reranked_file=${reranked_file/.json/-debug.json}
                reranked_output_dir=${reranked_output_dir}-debug
            fi
            print_and_run $reranked_file $reranked_output_dir
        done
    done
fi
else
    echo "Invalid algo"
    exit 1
fi

end_time=$(date +%s)
execution_time=$(expr $end_time - $start_time)

echo "Finish"
echo "Execution time: $execution_time seconds"