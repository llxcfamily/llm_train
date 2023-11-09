This is a script for llm train

###CUDA Version###
We sugget use nvidia cuda 12.1 in linux server:
  1. wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
  2. sudo sh cuda_12.1.0_530.30.02_linux.run
  you can get following messages with nvidia-smi 
  | NVIDIA-SMI 530.30.02              Driver Version: 530.30.02    CUDA Version: 12.1     |

###Download model ###
download model from https://huggingface.co/luodian/llama-7b-hf, then put it to models/llama-7b-hf


###for countinous pretrain###
1. train chinese spm model , then merge original llama tokenizer model and chinese-spm to get a new tokenizer model.
   `bash data_process/spm_train.sh`

2. replace models/llama-7b-hf/tokenizer.model with above merged tokenizer model.

3. bash run_pretrain.sh

###for sft###
-full parameters fine-tuning
`bash run_llama_sft.sh`

-lora finetuning
`bash run_llama_lora.sh`
