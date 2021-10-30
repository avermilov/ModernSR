import os

import matplotlib.pyplot as plt
import scipy.io as sio
from tqdm import tqdm

from utils.parser import get_visualize_kernels_parser

if __name__ == "__main__":
    argparser = get_visualize_kernels_parser()
    args = argparser.parse_args()

    kernels_path = args.ker_path
    results_path = args.res_path

    if kernels_path is None or results_path is None:
        raise ValueError("Both kernel source path and destination path must be specified.")

    kernel_names = []
    kernels = []
    for filename in tqdm(os.listdir(kernels_path), desc="Kernels"):
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
