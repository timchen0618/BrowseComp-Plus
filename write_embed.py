import json

template_file = "embed_template.SBATCH"
org_template = open(template_file, "r").read()
write_to_run_file = []

for model in ["infly", "qwen3-0.6b"]:
    for num in range(0, 16):
        
        template = org_template.replace("[model]", model).replace("[num]", str(num))
        write_to_run_file.append(f"sbatch sbatch_jobs/embed_{model}_{num}.SBATCH")
        with open(f"sbatch_jobs/embed_{model}_{num}.SBATCH", "w") as f:
            f.write(template)
            
            
with open("run_embed.sh", "w") as f:
    f.write("#!/bin/bash\n")
    f.write("\n".join(write_to_run_file))