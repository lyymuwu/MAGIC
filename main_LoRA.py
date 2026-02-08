from src.utils import GPU_Search
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_Search())

import numpy as np
import torch
import time
import sys

sys.path.append("src/")
from src.task_vectors import TaskVector
from src.args import parse_arguments
from src.utils import *
from src.utils_merge_func import *
from src.utils_rectify import *
from safetensors.torch import load_file, save_file

# Reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Parse CLI arguments
args = parse_arguments()
# args.merge = "WA"  # Optional override
# args.space = "W"   # Optional override

# Datasets used for evaluation/validation when required by merging method
exam_datasets = [
    "SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD",
]
val_datasets = ['ImageNet'] if args.agnostic else exam_datasets  # or 'CIFAR100Val'
train_datasets = exam_datasets
eval_datasets = exam_datasets
args.DATASETS = train_datasets

# Logging
str_time_ = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
log = create_log_dir(args.logs_path, "log_{}_clip.txt".format(str_time_))
log.info(
    "space: {}, merge: {}, req: {}, anchor: {}".format(
        args.space, args.merge, args.req, args.anchor
    )
)
log.info(
    "drop: {}, combine: {}, calibrate_norm: {}".format(
        args.drop, args.combine, args.norm
    )
)
log.info("TRIM_drop: {}, DARE_drop: {}".format(args.TRIM_drop, args.DARE_drop))
log.info(f"Alpha: {args.Alpha}")
starttime = time.time()

# ---------------- Task vectors (LoRA) ----------------
# Step 1: Put the LoRA safetensors under args.base_dir.
# Step 2: Load each LoRA as a TaskVector.
args.base_dir = "/data/Lora"
# Object LoRA
file_path = f"{args.base_dir}/Haunter_Pokemon_SDXL.safetensors"
model_object = load_file(file_path)
tv_object = TaskVector(vector=model_object)

# Style LoRA
file_path = f"{args.base_dir}/Anime_Sketch_SDXL.safetensors"
model_style = load_file(file_path)
tv_style = TaskVector(vector=model_style)

task_vectors = [tv_object, tv_style]

# Only W/N spaces are supported for LoRA merging
assert args.space in ['W', 'N'], "When merging LoRA, only W space and N space are supported."

# Merge and get the merged LoRA weights
task_vector_merged = merging(
    task_vectors, log, args, args.merge, args.space, val_datasets, return_tv=True
)
lora_merged = task_vector_merged.vector

# Save merged LoRA for Stable Diffusion WebUI
save_file(lora_merged, f"/data/stable-diffusion-webui/models/Lora/merged_{args.merge}_{args.space}.safetensors")
log.info("Merging finished!, named merged_{}_{}.safetensors".format(args.merge, args.space))