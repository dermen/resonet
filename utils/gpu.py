import torch

def get_mem(dev_id):
    # https://stackoverflow.com/a/58216793/2077270
    t = torch.cuda.get_device_properties(dev_id).total_memory
    r = torch.cuda.memory_reserved(dev_id)
    a = torch.cuda.memory_allocated(dev_id)
    f = r-a  # free inside reserved
    return t/1024**3,f/1024**3




