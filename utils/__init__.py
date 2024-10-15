import random
import os
from pathlib import Path
import numpy as np
import torch


def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
