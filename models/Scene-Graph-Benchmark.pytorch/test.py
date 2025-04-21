# test_cuda.py
import torch

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0))

x = torch.tensor([1.0, 2.0, 3.0]).cuda()
print("Tensor on CUDA:", x)
