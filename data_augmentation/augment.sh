# Example usage: ./augment.sh 1 20 0 0

split_start=$1
split_end=$2
cuda=$3
seed=$4

for split in $(seq $split_start $split_end)
do
    echo "Augmenting split $split on cuda $cuda"
    python augment_t5_batch.py --split $split --device $cuda --seed $seed
    echo "Done augmenting split $split on cuda $cuda"
done