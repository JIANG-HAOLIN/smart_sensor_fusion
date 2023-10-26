import torch


print(f"if Cuda available:{torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Cuda info:\n{torch.cuda.get_device_properties('cuda')}")
    print(f"Cuda version:{torch.version.cuda}")
else:
    print(f'no Cuda detected, using CPU instead !!')
