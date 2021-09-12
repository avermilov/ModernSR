import os
import scipy.io as sio
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm

args = ArgumentParser()
args.add_argument("--source_dir", type=str, default=None)
args.add_argument("--dest_dir", type=str, default=None)
args = args.parse_args()

kernels_path = args.ker_path
results_path = args.res_path

if kernels_path is None or results_path is None:
    raise ValueError("Both kernel source path and destination path must be specified.")

kernel_names = []
kernels = []
for filename in tqdm(os.listdir(kernels_path)):
    abs_path = os.path.join(kernels_path, filename)
    try:
        mat = sio.loadmat(abs_path)
        mat = mat["Kernel"]
        kernel_names.append(filename[:filename.rfind(".")] + ".png")
        kernels.append(mat)
    except Exception:
        print("Error while reading kernel file: " + abs_path)

for name, kernel in zip(kernel_names, kernels):
    plt.imsave(results_path + name, kernel)
