import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.3
capacity = 1.0
#task = "copa"
for task in ["openbookqa", "winogrande", "copa"]:
    ## H2O
    print("="*10+" H2O " + "="*10)
    # OPT
    for ratio in [0.25, 0.125, 0.0625, 0.03125]:
      print(f"{'*'*20}ratio={ratio}{'*'*20}")
      cmd = []
      cmd.append("bash h2o.sh")
      cmd.append(task)
      cmd.append(f"/home/liushenyu/opt-6.7b/")
      # cmd.append(f"facebook/opt-{size}")
      cmd.append("opt")
      cmd.append(str(shots))
      cmd.append(str(ratio)) # heavy_ratio
      cmd.append(str(ratio)) # recent_ratio
      cmd = ' '.join(cmd)
      print(cmd)
      os.system(cmd)
    print("-------------------------------------------")




