import torch
print(torch.cuda.is_available()) # If this is False, you have the wrong version.
print(torch.version.cuda)        # If this raises an error, CUDA support is missing entirely.
