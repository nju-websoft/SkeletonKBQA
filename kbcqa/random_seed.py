
import os
import torch
import numpy as np
import random


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # gpu
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


"""1 2 3 4 5 6 7 8 9"""
set_seed(0)
print(torch.rand(5))
