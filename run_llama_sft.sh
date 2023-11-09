# export CUDA_VISIBLE_DEVICES=0

TRAIN=./data/train_dummy.csv
VAL=./data/train_dummy.csv
OUTPUT=./models/llama_7b_full_ft

export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
GPUS_PER_NODE=2
NODE_RANK=0

log_path=./run_log/sft_1/
mkdir -p $log_path

torchrun --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --master_port $MASTER_PORT \
  --master_addr $MASTER_ADDR \
  ./code/run_llama.py \
  --model_name_or_path ./models/llama-7b-hf \
  --train_file $TRAIN \
  --max_seq_length 2048 \
  --output_dir $OUTPUT \
  --do_train \
  --do_eval False \
  --evaluation_strategy no \
  --eval_steps 1000000000 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --optim adamw_torch \
  --adam_beta2 0.95 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --num_train_epochs 3 \
  --lr_scheduler_type cosine \
  --logging_first_step True \
  --logging_steps 10 \
  --logging_nan_inf_filter False \
  --save_strategy steps \
  --save_steps 100 \
  --save_total_limit 2 \
  --fp16 True \
  --disable_tqdm False \
  --log_on_each_node False \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --overwrite_cache False \
  --overwrite_output_dir \
  --deepspeed ./code/zero_3.json  2>&1 | tee "$log_path/log_$RANK"


