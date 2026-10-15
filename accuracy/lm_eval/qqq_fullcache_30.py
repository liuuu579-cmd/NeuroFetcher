import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.2
capacity = 1.0


## neural
print("="*10+" neural " + "="*10)
# OPT
task = "winogrande"
cmd = []
cmd.append("bash neural.sh")
cmd.append(task)
cmd.append(f"/home/liushenyu/opt-13b/")
cmd.append(f"/home/liushenyu/opt-13b/")
cmd.append("opt")
cmd.append(str(shots))
cmd.append(f"/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth")
cmd.append("0.3")

cmd = ' '.join(cmd)
print(cmd)
os.system(cmd)
print("-------------------------------------------")

## Baseline
print("="*10+" Full cache " + "="*10)
# OPT
cmd = []
cmd.append("bash full_cache.sh")
cmd.append(task)
cmd.append(f"/home/liushenyu/opt-13b/")
cmd.append("opt")
cmd.append(str(shots))
cmd = ' '.join(cmd)
print(cmd)
os.system(cmd)
print("-------------------------------------------")
