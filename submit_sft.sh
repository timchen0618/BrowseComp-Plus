# # Filtered Commands

# # Single Random Selection (Single Input)
# sbatch --export=ALL,INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed2.jsonl,TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_seed0/,EVAL_FOLDER=evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_summary_orig_ext_selected_tools_random_seed2_gpt-oss-120b_seed0/,RUN_NAME=random_selection_seed2  sft_train.SBATCH
# sleep 200
# # Best of 4 Random Selection
# sbatch -J sft_best_of_8_random_selection_mode_d --export=ALL,MULTI_INPUT=sft/axolotl/multi_input_config_best_of_random_selection.json,MODE=d,RUN_NAME=best_of_8_random_selection_mode_d sft_train.SBATCH
# sleep 200
# # Gemini 2.5 Pro Selection (Single Input)
# sbatch -J sft_gemini-2.5-pro_selection_0 --export=ALL,INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl,TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_seed0/,EVAL_FOLDER=evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0/,RUN_NAME=gemini-2.5-pro_selection_0  sft_train.SBATCH
# sleep 200
# # Best of 4 Gemini Selection
# sbatch -J sft_best_of_4_gemini-2.5-pro_selection_mode_d --export=ALL,MULTI_INPUT=sft/axolotl/multi_input_config_best_of_gemini_selection.json,MODE=d,RUN_NAME=best_of_4_gemini-2.5-pro_selection_mode_d sft_train.SBATCH

# # Single GPT-OSS-120B Scout 
# sbatch -J sft_gpt-oss-120b_scout --export=ALL,INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_budget_seed0.jsonl,TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_seed0/,EVAL_FOLDER=evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_budget_orig_ext_gpt-oss-120b_seed0/,RUN_NAME=gpt-oss-120b_scout  sft_train.SBATCH


# # Unfiltered Commands
# sleep 200
# # Single Random Selection (Single Input, unfiltered)
# sbatch -J sft_random_selection_seed2_unfiltered --export=ALL,KEEP_FAILED=--keep-failed,INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_random_seed2.jsonl,TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_seed0/,EVAL_FOLDER=evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_summary_orig_ext_selected_tools_random_seed2_gpt-oss-120b_seed0/,RUN_NAME=random_selection_seed2_unfiltered  sft_train.SBATCH
# sleep 200
# # Best of 4 Random Selection (unfiltered)
# sbatch -J sft_best_of_8_random_selection_mode_d_unfiltered --export=ALL,KEEP_FAILED=--keep-failed,MULTI_INPUT=sft/axolotl/multi_input_config_best_of_random_selection.json,MODE=d,RUN_NAME=best_of_8_random_selection_mode_d_unfiltered sft_train.SBATCH
# sleep 200
# # Gemini 2.5 Pro Selection (Single Input, unfiltered)
# sbatch -J sft_gemini-2.5-pro_selection_0_unfiltered --export=ALL,KEEP_FAILED=--keep-failed,INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl,TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_seed0/,EVAL_FOLDER=evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0/,RUN_NAME=gemini-2.5-pro_selection_0_unfiltered  sft_train.SBATCH
# sleep 200
# # Best of 4 Gemini Selection
# sbatch -J sft_best_of_4_gemini-2.5-pro_selection_mode_d_unfiltered --export=ALL,KEEP_FAILED=--keep-failed,MULTI_INPUT=sft/axolotl/multi_input_config_best_of_gemini_selection.json,MODE=d,RUN_NAME=best_of_4_gemini-2.5-pro_selection_mode_d_unfiltered sft_train.SBATCH

# # Single GPT-OSS-120B Scout
# sbatch -J sft_gpt-oss-120b_scout_unfiltered --export=ALL,KEEP_FAILED=--keep-failed,INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_budget_seed0.jsonl,TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_seed0/,EVAL_FOLDER=evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_budget_orig_ext_gpt-oss-120b_seed0/,RUN_NAME=gpt-oss-120b_scout_unfiltered  sft_train.SBATCH



# Full Fine-Tuning Commands

# Gemini 2.5 Pro Selection (Single Input, full FT)
sbatch -J sft_gemini-2.5-pro_selection_0_full --export=ALL,CONFIG=sft/axolotl/qwen3.5_4b_search_sft_full.yaml,INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl,TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_seed0/,EVAL_FOLDER=evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0/,RUN_NAME=gemini-2.5-pro_selection_0_full  sft_train.SBATCH

sleep 200
# Gemini 2.5 Pro Selection (Single Input, full FT, unfiltered)
sbatch -J sft_gemini-2.5-pro_selection_0_full_unfiltered --export=ALL,KEEP_FAILED=--keep-failed,CONFIG=sft/axolotl/qwen3.5_4b_search_sft_full.yaml,INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_gpt-oss-120b_use_original_messages.jsonl,TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_seed0/,EVAL_FOLDER=evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_summary_orig_ext_selected_tools_gpt-oss-120b_seed0/,RUN_NAME=gemini-2.5-pro_selection_0_full_unfiltered  sft_train.SBATCH
sleep 200
# Single GPT-OSS-120B Scout (full FT)
sbatch -J sft_gpt-oss-120b_scout_full --export=ALL,CONFIG=sft/axolotl/qwen3.5_4b_search_sft_full.yaml,INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_budget_seed0.jsonl,TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_seed0/,EVAL_FOLDER=evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_budget_orig_ext_gpt-oss-120b_seed0/,RUN_NAME=gpt-oss-120b_scout_full  sft_train.SBATCH
sleep 200
# Single GPT-OSS-120B Scout (full FT, unfiltered)
sbatch -J sft_gpt-oss-120b_scout_full_unfiltered --export=ALL,KEEP_FAILED=--keep-failed,CONFIG=sft/axolotl/qwen3.5_4b_search_sft_full.yaml,INPUT=selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_budget_seed0.jsonl,TRAJECTORY_FOLDER=runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/budget5_seed0/,EVAL_FOLDER=evals/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/traj_budget_orig_ext_gpt-oss-120b_seed0/,RUN_NAME=gpt-oss-120b_scout_full_unfiltered  sft_train.SBATCH

# full finetuning session
# claude --resume d14b5197-c4eb-4595-a13a-8ec581bf3739