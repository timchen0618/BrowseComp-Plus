from huggingface_hub import snapshot_download
import os
import torch
from transformers import AutoModel
from transformers import AutoTokenizer
# snapshot_download(
#     repo_id="denizqian/diverse-retriever-models",
#     repo_type="model",
#     allow_patterns="*/checkpoint/best_model/*",
#     local_dir="/scratch/hc3337/models/iterative_retrieval",
#     local_dir_use_symlinks=False
# )


def load(base_model_name, dir_path):
    epoch_path = os.path.realpath(dir_path)
    checkpoint_path = os.path.join(epoch_path, "checkpoint.pth")
    print(f"loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"]
    # remove the prefix of the state_dict. Just strip one "encoder."
    # just remove one "encoder."
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("encoder.inf_model."):
            print('stripping encoder.inf_model.')
            new_state_dict[k[len("encoder.inf_model."):]] = v
        elif k.startswith("encoder.qwen_model."):
            print('stripping encoder.qwen_model.')
            new_state_dict[k[len("encoder.qwen_model."):]] = v
        elif k.startswith("encoder."):
            print('stripping encoder.')
            new_state_dict[k[len("encoder."):]] = v
        else:
            print('k is not starting with encoder.inf_model. or encoder.', k)
            new_state_dict[k] = v

    model = AutoModel.from_pretrained(base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    # model = None
    model.load_state_dict(new_state_dict, strict=True)

    return model, tokenizer

os.environ["TRITON_CACHE_DIR"] = "/scratch/hc3337/.triton/"

# for model_type in ['contriever', 'infly', 'qwen3']:
for model_type in ['qwen3']:
    epoch_path = f"/scratch/hc3337/models/iterative_retrieval/{model_type}-base/checkpoint/best_model/"
    
    if model_type == 'contriever':
        model, tokenizer = load('facebook/contriever-msmarco', epoch_path)
    elif model_type == 'infly':
        model, tokenizer = load('infly/inf-retriever-v1-1.5b', epoch_path)
    elif model_type == 'qwen3':
        model, tokenizer = load('Qwen/Qwen3-Embedding-0.6B', epoch_path)
        
    model.save_pretrained(f"/scratch/hc3337/models/iterative_retrieval/{model_type}-finetuned")
    tokenizer.save_pretrained(f"/scratch/hc3337/models/iterative_retrieval/{model_type}-finetuned")
    
    # save in huggingface, so later can be loaded with huggingface .from_pretrained()
    