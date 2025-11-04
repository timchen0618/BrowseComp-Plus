import json
from pathlib import Path
import csv

def read_jsonl(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def read_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def parse_output(file_path, id2query):
    data = read_json(file_path)
    query = id2query.get(data.get('query_id', ''))
    results = data.get('result', [])
    output = [query]
    for result in results:
        if result['type'] == "reasoning":
            output.append(result['output'])
        elif result['type'] == "tool_call":
            output.append(result['arguments'])
    return output

def write_csv(outputs):
    with open('outputs.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerows(outputs)

if __name__ == "__main__":
    raw_data = read_jsonl('data/browsecomp_plus_decrypted.jsonl')
    id2query = {record['query_id']: record['query'] for record in raw_data}
    
    rootdir = Path('runs/bm25/oss-20b/')
    outputs = []
    for file in rootdir.glob('*.json'):
        out_data = parse_output(file, id2query)
        outputs.append(out_data)
    write_csv(outputs)