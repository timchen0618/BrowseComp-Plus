PREVIEW_CHARS=1200
REASONING_MAX_CHARS=2000
TOOL_OUTPUT_MAX_CHARS=3000
CONTEXT_MAX_CHARS=280000

# # gpt-oss-120b (830 files in seed4)
# python src_select_tool_calls/select_useful_tool_calls.py --k 5 --use-original-messages \
#   --preview-chars ${PREVIEW_CHARS} \
#   --context-reasoning-max-chars ${REASONING_MAX_CHARS} \
#   --context-tool-max-chars ${TOOL_OUTPUT_MAX_CHARS} \
#   --context-max-chars ${CONTEXT_MAX_CHARS} \
#   --output selected_tool_calls/all/gpt-oss-120b/seed0/selected_tool_calls_gpt-oss-120b_use_original_messages_gemini_3.1-pro-preview.jsonl \
#   --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/gpt-oss-120b/seed4/ \
#   --num-threads 8 \
#   --model @vertexai-gemini-ec5413/gemini-3.1-pro-preview

# # tongyi (830 files in seed4)
# python src_select_tool_calls/select_useful_tool_calls_tongyi.py --k 5  \
#   --preview-chars ${PREVIEW_CHARS} \
#   --context-reasoning-max-chars ${REASONING_MAX_CHARS} \
#   --context-tool-max-chars ${TOOL_OUTPUT_MAX_CHARS} \
#   --context-max-chars ${CONTEXT_MAX_CHARS} \
#   --output selected_tool_calls/all/tongyi/seed0/selected_tool_calls_tongyi_use_original_messages.jsonl \
#   --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/tongyi/seed4/ \
#   --num-threads 8 

# glm
# python src_select_tool_calls/select_useful_tool_calls_glm.py --k 5  \
#   --preview-chars ${PREVIEW_CHARS} \
#   --context-reasoning-max-chars ${REASONING_MAX_CHARS} \
#   --context-tool-max-chars ${TOOL_OUTPUT_MAX_CHARS} \
#   --context-max-chars ${CONTEXT_MAX_CHARS} \
#   --output selected_tool_calls/all/glm/seed0/selected_tool_calls_glm_use_original_messages.jsonl \
#   --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/glm/seed0/ 

# # minimax (150 files in seed0)                                                                                                                                                                  
# python src_select_tool_calls/select_useful_tool_calls_glm.py  --k 5 \
#   --preview-chars ${PREVIEW_CHARS} \
#   --context-reasoning-max-chars ${REASONING_MAX_CHARS} \
#   --context-tool-max-chars ${TOOL_OUTPUT_MAX_CHARS} \
#   --context-max-chars ${CONTEXT_MAX_CHARS} \
#   --num-threads 8 \
#   --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/minimax/seed0 \
#   --output selected_tool_calls/all/minimax/seed0/selected_tool_calls_minimax_use_original_messages.jsonl \
#   --queries topics-qrels/bcp/queries_test150.tsv                                                                                                                                                
                                                                                                                                                                                                  
# # qwen3.5-122b-a10b (150 files in seed0)                                                                                                                                                        
# python src_select_tool_calls/select_useful_tool_calls_glm.py  --k 5 \
#   --preview-chars ${PREVIEW_CHARS} \
#   --context-reasoning-max-chars ${REASONING_MAX_CHARS} \
#   --context-tool-max-chars ${TOOL_OUTPUT_MAX_CHARS} \
#   --context-max-chars ${CONTEXT_MAX_CHARS} \
#   --num-threads 8 \
#   --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/qwen3.5-122b-a10b/seed0 \
#   --output selected_tool_calls/all/qwen3.5-122b-a10b/seed0/selected_tool_calls_qwen3.5-122b-a10b_use_original_messages.jsonl \
#   --queries topics-qrels/bcp/queries_test150.tsv  


# Random selection
# python src_select_tool_calls/random_select_tool_calls.py \                                                                                                                                      
#       --input-jsonl selected_tool_calls/all/tongyi/seed0/selected_tool_calls_tongyi_use_original_messages.jsonl \                                                                                 
#       --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/tongyi/seed4 \                                                                                                                            
#       --format tongyi                                                                                                                                                                             
                                                                                                                                                                                                  
#   python src_select_tool_calls/random_select_tool_calls.py \                                                                                                                                      
#       --input-jsonl selected_tool_calls/all/glm/seed0/selected_tool_calls_glm_use_original_messages.jsonl \                                                                                       
#       --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/glm/seed0 \                                                                                                                               
#       --format glm                                                                                                                                                                                
                                                                                                                                                                                                  
#   python src_select_tool_calls/random_select_tool_calls.py \                                                                                                                                      
#       --input-jsonl selected_tool_calls/all/minimax/seed0/selected_tool_calls_minimax_use_original_messages.jsonl \                                                                               
#       --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/minimax/seed0 \                                            
#       --format glm                                                                                                                                                                                
                                                                                                                                                                                                  
#   python src_select_tool_calls/random_select_tool_calls.py \
#       --input-jsonl selected_tool_calls/all/qwen3.5-122b-a10b/seed0/selected_tool_calls_qwen3.5-122b-a10b_use_original_messages.jsonl \                                                           
#       --trajectory-dir runs/bcp/Qwen3-Embedding-8B/full/qwen3.5-122b-a10b/seed0 \                                                      
#       --format glm 