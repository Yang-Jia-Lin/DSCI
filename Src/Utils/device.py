"""
Src/Utils/device.py
"""

import torch


def get_device(verbose: bool = True) -> torch.device:
    """
    检查 CUDA 是否可用，返回 torch.device，并可选地打印信息。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        if device.type == 'cuda':
            name = torch.cuda.get_device_name(torch.cuda.current_device())
            print(f'Using CUDA device: {name} \n')
        else:
            print('CUDA not available, using CPU. \n')
    return device