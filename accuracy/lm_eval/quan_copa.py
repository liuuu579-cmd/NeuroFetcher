import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.2
capacity = 1.0
task = "winogrande" 
qbits = 1
print(f"{'*'*20}qbits={qbits}{'*'*20}")
cmd = []
cmd.append("bash quant.sh")
cmd.append(task)
cmd.append(f"/home/liushenyu/opt-6.7b/")
cmd.append("opt")
cmd.append(str(shots))
cmd.append(str(qbits))
cmd = ' '.join(cmd)
print(cmd)
os.system(cmd)
#task = "copa"
for task in ["openbookqa", "winogrande", "copa"]:
    ## Quant.
    print("="*10+" Quantization " + "="*10)
    for qbits in [8, 4, 2, 1]:
      print(f"{'*'*20}qbits={qbits}{'*'*20}")
      cmd = []
      cmd.append("bash quant.sh")
      cmd.append(task)
      cmd.append(f"/home/liushenyu/opt-6.7b/")
      cmd.append("opt")
      cmd.append(str(shots))
      cmd.append(str(qbits))
      cmd = ' '.join(cmd)
      print(cmd)
      os.system(cmd)
    print("-------------------------------------------")