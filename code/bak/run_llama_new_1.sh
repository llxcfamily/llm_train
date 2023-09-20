#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=1,2,4,7
#export CUDA_VISIBLE_DEVICES=0
#export NCCL_SOCKET_IFNAME=ens8



TRAIN=/home/work/zhanglong/data/id.json
VAL=data/train_clean_q_a_eos_new_7.csv
OUTPUT=/home/work/zhanglong/models/llama_13b_alpaca_en_id_new_1

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
  ./code/run_llama_output_loss.py \
  --ddp_timeout 72000000 \
  --model_name_or_path /home/work/zhanglong/models/llama-13b-hf \
  --train_file $TRAIN \
  --validation_file $VAL \
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
  --save_strategy epoch \
  --save_steps 50 \
  --save_total_limit 2 \
  --fp16 True \
  --disable_tqdm False \
  --log_on_each_node False \
  --report_to tensorboard \
  --gradient_checkpointing True \
  --overwrite_cache False \
  --deepspeed ./code/zero_3.json  2>&1 | tee "$log_path/log_$RANK"

