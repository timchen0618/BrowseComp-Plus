data_dir="sft/axolotl/data/raw/best_of_4_random_selection_mode_c"
python sft/axolotl/prepare_dataset.py \
    --multi-input sft/axolotl/multi_input_config_best_of_random_selection.json \
    --mode c \
    --output-dir ${data_dir} \
    --template qwen-oss \
    --split bcp-train680-test150 \
    --seed 42
