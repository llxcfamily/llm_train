# export CUDA_VISIBLE_DEVICES=0,1,2,3
# NUM_GPU=1
# PORT_ID=7777

TRAIN=./data/alpaca_data.json
# TRAIN=./data/sharegpt_clean.csv
OUTPUT=./models/llama_13b_sft_zero3

# torchrun \
#   --nproc_per_node $NUM_GPU \
#   --master_port $PORT_ID \


MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
NNODES=$WORLD_SIZE
GPUS_PER_NODE=1
NODE_RANK=$RANK

log_path=./run_log/$AIP_RUN_ID
mkdir -p $log_path

torchrun \
  --nproc_per_node $GPUS_PER_NODE \
  --nnodes $NNODES \
  --node_rank $NODE_RANK \
  --master_port $MASTER_PORT \
  --master_addr $MASTER_ADDR \
  ./code/run_llama_test.py \
  --model_name_or_path ./models/llama-13b-hf \
  --train_file $TRAIN \
  --max_seq_length 2048 \
  --output_dir $OUTPUT \
  --do_train \
  --do_eval False \
  --evaluation_strategy no \
  --eval_steps 1000000000 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
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
  --deepspeed ./code/zero_3.json 2>&1 | tee "$log_path/log_$RANK"


