import json

command = 'run_qwen3'

if command == 'gen_embed':
    template_file = "embed_template.SBATCH"
    org_template = open(template_file, "r").read()
    write_to_run_file = []

    # for model in ["infly", "qwen3-0.6b"]:
    for model in ["contriever"]:
        for num in range(0, 16):
            
            template = org_template.replace("[model]", model).replace("[num]", str(num))
            write_to_run_file.append(f"sbatch sbatch_jobs/embed_{model}_{num}.SBATCH")
            with open(f"sbatch_jobs/embed_{model}_{num}.SBATCH", "w") as f:
                f.write(template)
                
                
    with open("run_embed.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("\n".join(write_to_run_file))

elif command == 'run_agentic':
    write_to_run_file = []
    # for model in ["infly", "qwen3-0.6b"]:
    for model in ["contriever"]:
        template_file = f"qampari_{model}.SBATCH"
        org_template = open(template_file, "r").read()
        for num in range(0, 10):
            template = org_template.replace("[num]", str(num))
            write_to_run_file.append(f"sbatch sbatch_jobs/qampari_{model}_{num}.SBATCH")
            with open(f"sbatch_jobs/qampari_{model}_{num}.SBATCH", "w") as f:
                f.write(template)
                
    with open("run_qampari.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("\n".join(write_to_run_file))
        
elif command == 'run_qwen3':
    write_to_run_file = []

    # for model in ["infly", "qwen3-0.6b"]:
    for _web in ["_web", ""]:
        template_file = f"run{_web}_qwen3.SBATCH"
        org_template = open(template_file, "r").read()
        for num in range(0, 10):
            
            template = org_template.replace("[num]", str(num))
            write_to_run_file.append(f"sbatch sbatch_jobs/run{_web}_qwen3_{num}.SBATCH")
            with open(f"sbatch_jobs/run{_web}_qwen3_{num}.SBATCH", "w") as f:
                f.write(template)
                
                
    with open(f"run.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("\n".join(write_to_run_file))