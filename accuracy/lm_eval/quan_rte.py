import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.3
capacity = 1.0
task = "rte"
size = "13b"


 ## quantization
print("="*10+" Quantization " + "="*10)
# OPT

for qbits in [8, 4, 2, 1]:
    print("-"*20+ str(qbits) + "-"*20)
    print("-"*20+ str(qbits) + "-"*20)
    cmd = []
    cmd.append("bash quant.sh")
    cmd.append(task)
    cmd.append(f"facebook/opt-{size}")
    cmd.append("opt")
    cmd.append(str(shots))
    cmd.append(str(qbits))
    cmd = ' '.join(cmd)
    print(cmd)
    os.system(cmd)
    print("-------------------------------------------")
