import argparse

import random

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import subprocess
# 顺序运行，确保 main.py 完全执行完才执行 main_d.py
print("Running main.py...")
subprocess.run(['python', 'main.py'], check=True)  # 加上 check=True 表示执行失败时会抛异常

print("main.py finished.\nRunning main_d.py...")
subprocess.run(['python', 'main_alignment.py'], check=True)

print("All scripts finished.")