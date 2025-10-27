import torch

print('='*60)
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
print('CUDA Version:', torch.version.cuda)
print('GPU Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)

if torch.cuda.is_available():
    print('GPU Name:', torch.cuda.get_device_name(0))
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print('GPU Memory:', round(gpu_mem, 2), 'GB')
else:
    print('GPU: Not available - running on CPU')

print('='*60)
