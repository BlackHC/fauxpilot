#!/bin/bash

MODEL=${1}
NUM_GPUS=${2}

echo "Converting model ${MODEL} with ${NUM_GPUS} GPUs"

# Check whether the model starts with "codegen" or "galactica"
if [[ ${MODEL} == codegen* ]]; then
  MODEL_TYPE="codegen"
elif [[ ${MODEL} == galactica* ]]; then
  MODEL_TYPE="galactica"
else
  echo "Unknown model type: ${MODEL}"
  exit 1
fi

echo ${MODEL_TYPE}

export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub

# If the model type is codegen, then we need to download the codegen and convert it:
if [[ ${MODEL_TYPE} == codegen ]]; then
    exit
    cp -r models/${MODEL}-${NUM_GPUS}gpu /models
    python3 codegen_gptj_convert.py --code_model Salesforce/${MODEL} ${MODEL}-hf
    python3 huggingface_gptj_convert.py -in_file ${MODEL}-hf -saved_dir /models/${MODEL}-${NUM_GPUS}gpu/fastertransformer/1 -infer_gpu_num ${NUM_GPUS}
    rm -rf ${MODEL}-hf
elif [[ ${MODEL_TYPE} == galactica ]]; then
    python3 huggingface_opt_convert.py -in_file facebook/${MODEL} -saved_dir /models/${MODEL}-${NUM_GPUS}gpu/fastertransformer/1 -infer_gpu_num ${NUM_GPUS} -processes 30 -weight_data_type fp16
fi


