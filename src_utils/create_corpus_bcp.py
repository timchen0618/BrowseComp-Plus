from datasets import load_dataset
import json

def write_jsonl(outputs, output_file):
    with open(output_file, 'w') as f:
        for item in outputs:
            f.write(json.dumps(item) + '\n')
            
ds = load_dataset("Tevatron/browsecomp-plus-corpus", split="train")


outputs = []

for item in ds:
    text = item["text"]
    outputs.append({"id": item["docid"], "contents": text})
    
    
write_jsonl(outputs, "data/browsecomp_plus_corpus.jsonl")