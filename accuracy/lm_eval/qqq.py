import json
import os

with open("figure11-config.json") as f:
    config = json.load(f)
os.system("mkdir -p results")

shots = 5
partial = 0.2
capacity = 1.0
task = "copa"
size = "13b"


# neural
print("="*10+" neural " + "="*10)
# OPT

cmd = []
cmd.append("bash neural.sh")
cmd.append(task)
cmd.append(f"/home/liushenyu/opt-{size}/")
cmd.append(f"/home/liushenyu/opt-{size}/")
cmd.append("opt")
cmd.append(str(shots))
#cmd.append(f"/home/liushenyu/test/predictor_topk_0.3_iwl_epoch_20.pth")
cmd.append(f"/home/liushenyu/test/predictor_alpha_2_dyn_posw_epoch_20.pth")
cmd.append("0.3")

cmd = ' '.join(cmd)
print(cmd)
os.system(cmd)
print("-------------------------------------------")


# ## Baseline
# print("="*10+" Full cache " + "="*10)
# # OPT
# cmd = []
# cmd.append("bash full_cache.sh")
# cmd.append(task)
# cmd.append(f"/home/liushenyu/opt-13b/")
# cmd.append("opt")
# cmd.append(str(shots))
# cmd = ' '.join(cmd)
# print(cmd)
# os.system(cmd)
# print("-------------------------------------------")



# # InfiniGen
# print("="*10+" InfiniGen " + "="*10)
# alpha, budget = config[f"opt-{size}"][task][2]
# cmd = []
# cmd.append("bash infinigen.sh")
# cmd.append(task)
# cmd.append(f"../setup/opt-model/opt-{size}")
# cmd.append(f"/home/liushenyu/opt-{size}/")
# cmd.append("opt")
# cmd.append(str(shots))
# cmd.append(str(partial))
# cmd.append(str(alpha))
# cmd.append(str(capacity))
# cmd.append(str(budget))
# cmd.append("1")
# cmd.append("1")
# cmd = ' '.join(cmd)
# print(cmd)
# os.system(cmd)
# print("-------------------------------------------")


# for i in range(5,11):
#     cmd = []
#     cmd.append("bash infinigen.sh")
#     cmd.append(task)
#     cmd.append(f"../setup/opt-model/opt-{size}")
#     cmd.append(f"/home/liushenyu/opt-{size}/")
#     cmd.append("opt")
#     cmd.append(str(shots))
#     cmd.append(str(partial))
#     cmd.append(str(alpha))
#     cmd.append(str(capacity))
#     cmd.append(str(budget))
#
#     # 这里将原本的 "11" 替换为循环变量 i，并转为字符串
#     cmd.append(str(2*i))
#
#     cmd.append("50")
#
#     # 拼接并执行命令
#     cmd_str = ' '.join(cmd)
#     print(f"Running loop index: {i}")
#     print(cmd_str)
#     os.system(cmd_str)
#     print("-------------------------------------------")


# ## H2O
# print("="*10+" H2O " + "="*10)
# # OPT
#
# cmd = []
# cmd.append("bash h2o.sh")
# cmd.append(task)
# cmd.append(f"/home/liushenyu/opt-13b/")
# cmd.append("opt")
# cmd.append(str(shots))
# cmd.append(str(0.25)) # heavy_ratio
# cmd.append(str(0.25)) # recent_ratio
# cmd = ' '.join(cmd)
# print(cmd)
# os.system(cmd)
# print("-------------------------------------------")


# ## Quant.
# print("="*10+" Quantization " + "="*10)
#
# cmd = []
# cmd.append("bash quant.sh")
# cmd.append(task)
# cmd.append(f"/home/liushenyu/opt-13b/")
# cmd.append("opt")
# cmd.append(str(shots))
# cmd.append(str(8))
# cmd = ' '.join(cmd)
# print(cmd)
# os.system(cmd)
# print("-------------------------------------------")


