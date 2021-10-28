import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PSNR_LOG_NAME = "Validation Metrics/Epoch Wise/PSNR"
LPIPS_ALEX_LOG_NAME = "Validation Metrics/Epoch Wise/LPIPS Alex"
LPIPS_VGG_LOG_NAME = "Validation Metrics/Epoch Wise/LPIPS VGG"
