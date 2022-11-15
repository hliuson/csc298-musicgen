#print data about available GPUs, CPUs, RAM, etc.
import torch
import psutil
import os

print("Available GPUs: ", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(" - GPU ", i, ": ", torch.cuda.get_device_name(i))
print("Available CPUs: ", psutil.cpu_count())
print("Total RAM (GB): ", psutil.virtual_memory().total / 1024 / 1024 / 1024)
print("Available RAM: ", psutil.virtual_memory().available / 1024 / 1024 / 1024)