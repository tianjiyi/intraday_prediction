import torch
print('Torch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
# print('Arch list:', torch.backends.cuda.get_arch_list())

# save as quick_gpu_check.py
import torch, time
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0), "CC:", torch.cuda.get_device_capability(0))

torch.cuda.synchronize()
x = torch.rand(8192, 8192, device="cuda", dtype=torch.float16)
t0 = time.time()
y = x @ x
torch.cuda.synchronize()
print("Matmul time (8192x8192, fp16):", round(time.time()-t0, 3), "s", "Device:", y.device)
