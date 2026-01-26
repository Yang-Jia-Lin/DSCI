"""
Src/Utils/utils_function.py
"""
import os
import platform
import subprocess
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    用于解决 JSON 无法直接序列化 Numpy 数据类型的问题
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            # 如果数组太长，只保存形状信息，保持 log 简洁
            if obj.size > 20:
                return f"<numpy_array shape={obj.shape} dtype={obj.dtype}>"
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def open_file(file_path):
    """跨平台打开文件"""
    if platform.system() == "Windows":
        os.startfile(file_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", str(file_path)])
    else:  # Linux
        subprocess.run(["xdg-open", str(file_path)])

