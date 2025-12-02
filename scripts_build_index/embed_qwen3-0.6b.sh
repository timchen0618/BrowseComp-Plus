model_name="Qwen/Qwen3-Embedding-0.6B"
corpus_name=$1

CUDA_VISIBLE_DEVICES=0 python -m tevatron.retriever.driver.encode \
  --model_name_or_path ${model_name} \
  --dataset_name csv \
  --dataset_path /scratch/hc3337/wikipedia_chunks/${corpus_name}.csv \
  --encode_output_path embeddings/qwen3-0.6b/${corpus_name}.pkl \
  --passage_max_len 4096 \
  --normalize \
  --pooling eos \
  --passage_prefix "" \
  --per_device_eval_batch_size 32 \
  --fp16