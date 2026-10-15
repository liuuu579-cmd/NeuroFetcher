import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.2
capacity = 1.0


task = "rte"
size = "13b"
# InfiniGen
print("="*10+" InfiniGen " + "="*10)
for retain_ratio in range(4):
  
  alpha, budget = config[f"opt-{size}"][task][retain_ratio]
  print(f"{'*'*10}alpha={alpha} budget={budget}{'*'*10}")
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


