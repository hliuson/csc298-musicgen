import torch
#detect gpu
print(torch.__version__)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using the GPU")
else:
    device = torch.device("cpu")
    print("Using the CPU")