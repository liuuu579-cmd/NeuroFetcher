import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.2
capacity = 1.0

size = "13b"




for task in ["openbookqa", "winogrande", "copa"]:
    ## Baseline
    print("="*10+" Full cache " + "="*10)
    # OPT
    cmd = []
    cmd.append("bash full_cache.sh")
    cmd.append(task)
    cmd.append(f"/home/liushenyu/opt-6.7b/")
    cmd.append("opt")
    cmd.append(str(shots))
    cmd = ' '.join(cmd)
    print(cmd)
    os.system(cmd)
    print("-------------------------------------------")
