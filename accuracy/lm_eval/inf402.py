import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.3
capacity = 1.0
alpha = 4
budget = 0.2
task = "winogrande"
size = "13b"
# InfiniGen

cmd = []
cmd.append("bash infinigen.sh")
cmd.append(task)
cmd.append(f"../setup/opt-model/opt-{size}")
cmd.append(f"/home/liushenyu/opt-13b/")
cmd.append("opt")
cmd.append(str(shots))
cmd.append(str(partial))
cmd.append(str(alpha))
cmd.append(str(capacity))
cmd.append(str(budget))
cmd = ' '.join(cmd)
print(cmd)
os.system(cmd)
  
print("-------------------------------------------")


