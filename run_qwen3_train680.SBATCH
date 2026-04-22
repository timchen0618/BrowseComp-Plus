#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=gpt-oss-120b_train680_seed0
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output=sbatch_outputs/gpt-oss-120b_train680_seed0.out
#SBATCH --account=torch_pr_152_courant
#SBATCH --constraint="h200"
#SBATCH --gres=gpu:1
#SBATCH --array=0-7

SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/bcp.ext3
VLLM_OVERLAY_FILE=${OVERLAY_FILE}
GPU_UTIL="0.85"
MODEL_NAME="gpt-oss-120b"
TENSOR_PARALLEL_SIZE=1
port=6011

PLAN_MODEL="gemini_2.5_pro"
TRAJ_MODEL="${MODEL_NAME}"

dataset="bcp"   # bcp only (train680 split is bcp-specific)
mode="org"
seed=0
SEARCH_BUDGET=5

lora_modules=""
if [ "$MODEL_NAME" == "tongyi" ]; then
    model_path="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
    PORT_STR="--port ${port}"
elif [ "$MODEL_NAME" == "qwen3.5-35b-a3b" ]; then
    model_path="Qwen/Qwen3.5-35B-A3B"
    PORT_STR=""
    VLLM_OVERLAY_FILE=/scratch/hc3337/envs/vllm_latest.ext3
elif [ "$MODEL_NAME" == "qwen3.6-35b-a3b" ]; then
    model_path="Qwen/Qwen3.6-35B-A3B"
    PORT_STR=""
    VLLM_OVERLAY_FILE=/scratch/hc3337/envs/vllm_latest.ext3
elif [ "$MODEL_NAME" == "qwen3.5-4b" ]; then
    model_path="Qwen/Qwen3.5-4B"
    PORT_STR=""
    VLLM_OVERLAY_FILE=/scratch/hc3337/envs/vllm_latest.ext3
elif [ "$MODEL_NAME" == "qwen3.5-4b-sft" ]; then
    model_path="Qwen/Qwen3.5-4B"
    lora_modules="qwen3.5-4b-sft=sft/checkpoints/axolotl-qwen3-4b-search"
    PORT_STR=""
    VLLM_OVERLAY_FILE=/scratch/hc3337/envs/vllm_latest.ext3
elif [ "$MODEL_NAME" == "gpt-oss-120b" ]; then
    model_path="openai/gpt-oss-120b"
    TENSOR_PARALLEL_SIZE=1
    GPU_UTIL="0.85"
    PORT_STR=""
elif [ "$MODEL_NAME" == "gpt-oss-20b" ]; then
    model_path="openai/gpt-oss-20b"
    PORT_STR=""
else
    echo "Invalid retriever name"
    exit 1
fi
client_model="${model_path}"
[ -n "${lora_modules}" ] && client_model="${MODEL_NAME}"


if [ "$MODEL_NAME" == "gpt-oss-120b" ] || [ "$MODEL_NAME" == "gpt-oss-20b" ]; then
    singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/BrowseComp-Plus/; HF_TOKEN=${HF_TOKEN} vllm serve ${model_path} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --gpu-memory-utilization ${GPU_UTIL}" &
elif [ "$MODEL_NAME" == "qwen3.5-35b-a3b" ] || [ "$MODEL_NAME" == "qwen3.6-35b-a3b" ] || [ "$MODEL_NAME" == "qwen3.5-4b" ]; then
    singularity exec --nv --overlay ${VLLM_OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/BrowseComp-Plus/; HF_TOKEN=${HF_TOKEN} VLLM_FLASH_ATTN_VERSION=2 CUDA_LAUNCH_BLOCKING=1 vllm serve ${model_path} --port ${port} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --gpu-memory-utilization ${GPU_UTIL} --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder" &
elif [ "$MODEL_NAME" == "qwen3.5-4b-sft" ]; then
    singularity exec --nv --overlay ${VLLM_OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/BrowseComp-Plus/; HF_TOKEN=${HF_TOKEN} VLLM_FLASH_ATTN_VERSION=2 CUDA_LAUNCH_BLOCKING=1 vllm serve ${model_path} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --gpu-memory-utilization ${GPU_UTIL} --enable-lora --lora-modules ${lora_modules} --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder" &
else
    singularity exec --nv --overlay ${VLLM_OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/BrowseComp-Plus/; HF_TOKEN=${HF_TOKEN} VLLM_FLASH_ATTN_VERSION=2 CUDA_LAUNCH_BLOCKING=1 vllm serve ${model_path} --port ${port} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} --gpu-memory-utilization ${GPU_UTIL}" &
fi

sleep 300

if [ "$MODEL_NAME" == "tongyi" ]; then
    execution_script="tongyi_client.py"
else
    execution_script="oss_client.py"
fi

RETRIEVER="Qwen3-Embedding-8B"
INDEX_PATH="indexes/Qwen3-Embedding-8B/corpus.shard*.pkl"
EMBED_MODEL_NAME="Qwen/Qwen3-Embedding-8B"

OUT_BASE="runs/${dataset}/${RETRIEVER}/train680/${MODEL_NAME}"

# Strip modifiers, setting flags along the way
base_mode="${mode%_reinject_every_5}"
[ "$base_mode" != "$mode" ] && reinject_args="--plan-reinject-every 5" || reinject_args=""

RANDOM_FILE_SUFFIX=""
# Strip trigger modifier
if [[ "$base_mode" == *_traj_ext ]]; then
    inner_mode="${base_mode%_traj_ext}"
    trigger="traj_ext"
elif [[ "$base_mode" == "traj_ext" ]]; then
    inner_mode="${base_mode}"
    trigger="traj_ext"
elif [[ "$base_mode" == "traj_summary_ext_selected_tools" ]]; then
    inner_mode="${base_mode}"
    trigger="traj_summary_ext"
elif [[ "$base_mode" == "traj_summary_ext" ]]; then
    inner_mode="${base_mode}"
    trigger="traj_summary_ext"
elif [[ "$base_mode" == "traj_orig_ext" ]]; then
    inner_mode="${base_mode}"
    trigger="traj_orig_ext"
elif [[ "$base_mode" == "traj_budget_orig_ext" ]]; then
    inner_mode="${base_mode}"
    trigger="traj_orig_ext"
elif [[ "$base_mode" == "traj_summary_orig_ext" ]]; then
    inner_mode="${base_mode}"
    trigger="traj_summary_orig_ext"
elif [[ "$base_mode" == "traj_summary_orig_ext_selected_tools" ]]; then
    inner_mode="${base_mode}"
    trigger="traj_summary_orig_ext_selected_tools"
elif [[ "$base_mode" == traj_summary_orig_ext_selected_tools_random_seed* ]]; then
    inner_mode="${base_mode}"
    trigger="traj_summary_orig_ext_selected_tools"
    RANDOM_FILE_SUFFIX="${base_mode#traj_summary_orig_ext_selected_tools}"
elif [[ "$base_mode" == *_start_ext ]]; then
    inner_mode="${base_mode%_start_ext}"
    trigger="start_ext"
elif [[ "$base_mode" == *_start_and_after_steps_5 ]]; then
    inner_mode="${base_mode%_start_and_after_steps_5}"
    trigger="start_and_after_steps"
elif [[ "$base_mode" == *_after_steps_5 ]]; then
    inner_mode="${base_mode%_after_steps_5}"
    trigger="after_steps"
else
    inner_mode="$base_mode"
    trigger="start"
fi

version="${inner_mode#planning_}"   # e.g. v1, v2, v0.5, v3, v4

# Default output dir — overridden only for org and start_ext
output_dir="${OUT_BASE}/${mode}_seed${seed}"

case "$inner_mode" in
    org)
        output_dir="${OUT_BASE}/seed${seed}"
        extra_args="--save-raw-messages"
        ;;
    planning_retrospective)
        extra_args="--planning-trigger start_ext --planning-file retrospective_plans/${MODEL_NAME}_seed0_gemini_legacy_${SLURM_ARRAY_TASK_ID}.jsonl"
        ;;
    traj_ext)
        output_dir="${OUT_BASE}/traj_ext_${TRAJ_MODEL}_seed${seed}"
        extra_args="--planning-trigger traj_ext --trajectory-dir runs/${dataset}/${RETRIEVER}/train680/${TRAJ_MODEL}/seed${seed} --query-template QUERY_TEMPLATE_GIVEN_TRAJECTORY"
        ;;
    traj_orig_ext)
        output_dir="${OUT_BASE}/traj_orig_ext_${TRAJ_MODEL}_seed${seed}"
        extra_args="--planning-trigger traj_orig_ext --trajectory-dir runs/${dataset}/${RETRIEVER}/train680/${TRAJ_MODEL}/seed$((seed + 4)) --query-template QUERY_TEMPLATE_GIVEN_TRAJECTORY"
        ;;
    traj_budget_orig_ext)
        output_dir="${OUT_BASE}/traj_budget_orig_ext_${TRAJ_MODEL}_seed${seed}"
        extra_args="--planning-trigger traj_orig_ext --trajectory-dir runs/${dataset}/${RETRIEVER}/train680/${TRAJ_MODEL}/budget${SEARCH_BUDGET}_seed${seed} --query-template QUERY_TEMPLATE_GIVEN_TRAJECTORY"
        ;;
    traj_summary_ext_selected_tools)
        output_dir="${OUT_BASE}/traj_summary_ext_selected_tools_${TRAJ_MODEL}_seed${seed}"
        extra_args="--planning-trigger traj_summary_ext --trajectory-summary-file selected_tool_calls/train680/gpt-oss-120b/seed${seed}/selected_tool_calls_train680_${SLURM_ARRAY_TASK_ID}.jsonl --query-template QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY"
        ;;
    traj_summary_ext)
        output_dir="${OUT_BASE}/traj_summary_ext_${TRAJ_MODEL}_seed${seed}"
        extra_args="--planning-trigger traj_summary_ext --trajectory-dir runs/${dataset}/${RETRIEVER}/train680/${TRAJ_MODEL}/seed${seed} --query-template QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY"
        ;;
    traj_summary_orig_ext)
        output_dir="${OUT_BASE}/traj_summary_orig_ext_${TRAJ_MODEL}_seed${seed}"
        extra_args="--planning-trigger traj_summary_orig_ext --trajectory-dir runs/${dataset}/${RETRIEVER}/train680/${TRAJ_MODEL}/seed$((seed + 4)) --query-template QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY"
        ;;
    traj_summary_orig_ext_selected_tools)
        output_dir="${OUT_BASE}/traj_summary_orig_ext_selected_tools_${TRAJ_MODEL}_seed${seed}"
        extra_args="--planning-trigger traj_summary_ext --trajectory-summary-file selected_tool_calls/train680/gpt-oss-120b/seed${seed}/selected_tool_calls_gpt-oss-120b_use_original_messages_train680_${SLURM_ARRAY_TASK_ID}.jsonl --query-template QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY"
        ;;
    traj_summary_orig_ext_selected_tools_random_seed*)
        output_dir="${OUT_BASE}/traj_summary_orig_ext_selected_tools${RANDOM_FILE_SUFFIX}_${TRAJ_MODEL}_seed${seed}"
        extra_args="--planning-trigger traj_summary_ext --trajectory-summary-file selected_tool_calls/train680/gpt-oss-120b/seed${seed}/selected_tool_calls_gpt-oss-120b_use_original_messages${RANDOM_FILE_SUFFIX}_train680_${SLURM_ARRAY_TASK_ID}.jsonl --query-template QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY"
        ;;
    planning_v*)
        if [ "$trigger" = "start_ext" ]; then
            reinject_suffix="${mode##*_start_ext}"   # "" or "_reinject_every_5"
            output_dir="${OUT_BASE}/${inner_mode}_start_ext_${PLAN_MODEL}${reinject_suffix}_seed${seed}"
            extra_args="--planning-trigger start_ext --planning-file output_plans/${PLAN_MODEL}/${dataset}/${PLAN_MODEL}_plans_${version}_train680_${SLURM_ARRAY_TASK_ID}.jsonl"
        else
            extra_args="--planning-trigger ${trigger} --plan-prompt-file prompts/planning_prompt_${version}.md"
        fi
        ;;
    budget)
        output_dir="${OUT_BASE}/budget${SEARCH_BUDGET}_seed${seed}"
        extra_args="--search-budget ${SEARCH_BUDGET} --save-raw-messages"
        ;;
    *)
        echo "Invalid mode name"
        exit 1
        ;;
esac

# --query-template is only supported by oss_client.py
if [ "$execution_script" != "oss_client.py" ]; then
    extra_args="${extra_args/ --query-template QUERY_TEMPLATE_GIVEN_TRAJECTORY/}"
    extra_args="${extra_args/ --query-template QUERY_TEMPLATE_GIVEN_TRAJ_SUMMARY/}"
fi

echo "Executing script: ${execution_script} | mode: ${mode} | output: ${output_dir}"
echo "version: ${version} | trigger: ${trigger} | inner_mode: ${inner_mode}"
echo "extra_args: ${extra_args}"
singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; cd /scratch/hc3337/projects/BrowseComp-Plus/; HF_TOKEN=${HF_TOKEN} CUDA_LAUNCH_BLOCKING=1 python search_agent/${execution_script} --output-dir ${output_dir} --searcher-type faiss --index-path '${INDEX_PATH}' --model-name ${EMBED_MODEL_NAME} --normalize --num-threads 10 --query topics-qrels/${dataset}/bcp_train680_8_shards/q_${SLURM_ARRAY_TASK_ID}.tsv --model ${client_model} ${PORT_STR} ${extra_args} ${reinject_args}"
