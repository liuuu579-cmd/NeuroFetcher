import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.2
capacity = 1.0
task = "openbookqa"
size = "13b"


 ## neural
print("="*10+" neural " + "="*10)
# OPT
for qbits in [0.6,0.3,0.15,0.08]:
  cmd = []
  cmd.append("bash neural.sh")
  cmd.append(task)
  cmd.append(f"/home/liushenyu/opt-13b/")
  cmd.append(f"/home/liushenyu/opt-13b/")
  cmd.append("opt")
  cmd.append(str(shots))
  #cmd.append(f"/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth")
  cmd.append(f"/home/liushenyu/test/predictor_alpha_2_dyn_posw_epoch_20.pth")
  cmd.append(str(qbits))
#
  cmd = ' '.join(cmd)
  print(cmd)
  os.system(cmd)
print("-------------------------------------------")
