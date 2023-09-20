#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=1,2,4,7
export CUDA_VISIBLE_DEVICES=0
#export NCCL_SOCKET_IFNAME=ens8


NUM_GPU=1
PORT_ID=7779

TRAIN=./data/sharegpt_clean_test.csv
VAL=data/train_clean_q_a_eos_new_7.csv
OUTPUT=./models/multilingual_sharegpt_v1.0

log_path=./run_log/$AIP_RUN_ID
mkdir -p $log_path

torchrun \
  --nproc_per_node $NUM_GPU \
  --master_port $PORT_ID \
  ./code/run_llama_new.py \
  --ddp_timeout 72000000 \
  --model_name_or_path ./models/llama-7b-hf \
  --train_file $TRAIN \
  --validation_file $VAL \
  --max_seq_length 2048 \
  --output_dir $OUTPUT \
  --do_train \
  --do_eval False \
  --evaluation_strategy no \
  --eval_steps 1000000000 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --optim adamw_torch \
  --adam_beta2 0.95 \
  --weight_decay 0.1 \
  --warmup_ratio 0.1 \
  --num_train_epochs 3 \
  --lr_scheduler_type constant_with_warmup \
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
  --deepspeed ./code/zero_2.json 2>&1 
  
  #| tee "$log_path/log_$RANK"

