#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=select_tools
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hc3337@nyu.edu
#SBATCH --output=sbatch_outputs/select_tools.out
#SBATCH --account=torch_pr_152_courant

SINGULARITY_IMAGE=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVERLAY_FILE=/scratch/hc3337/envs/qwen3.5.ext3

# seed=${SLURM_ARRAY_TASK_ID}
# MODEL_NAME="gpt-oss-120b"
# CMD_PREFIX="source /ext3/env.sh; conda activate qwen3.5; cd /scratch/hc3337/projects/BrowseComp-Plus/; python select_useful_tool_calls.py"
# TRAJ_DIR="runs/bcp/Qwen3-Embedding-8B/full/${MODEL_NAME}/seed$((seed + 4))"
# OUTPUT_DIR="selected_tool_calls/full/${MODEL_NAME}/seed${seed}"
# OUTPUT_FILE="selected_tool_calls_${MODEL_NAME}_use_original_messages.jsonl"
# NUM_THREADS=8
# singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "${CMD_PREFIX} --trajectory-dir ${TRAJ_DIR} --output ${OUTPUT_DIR}/${OUTPUT_FILE} --use-original-messages --num-threads ${NUM_THREADS}"

singularity exec --nv --overlay ${OVERLAY_FILE}:ro $SINGULARITY_IMAGE /bin/bash -c "source /ext3/env.sh; conda activate qwen3.5; cd /scratch/hc3337/projects/BrowseComp-Plus/; bash src_select_tool_calls/select.sh"