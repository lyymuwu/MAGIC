from src.utils import GPU_Search
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_Search())

import numpy as np
import torch
import time
import sys

sys.path.append("src/")
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments
from src.utils import *
from src.utils_merge_func import *
from src.utils_rectify import *


seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)



################################################################################
args = parse_arguments()
args.base_dir = "."
args.model = "BERT"
args.logs_path = "logs/" + args.model
pretrained_checkpoint = os.path.join(args.base_dir, "checkpoints", args.model, "pretrained_model.pth")

args.NLP = True
args.scaling_coef = 1
# args.merge = (
#     "TSV-M"  # TIES | DARE | TA | SLERP | Consensus_TA | ISO-C | ISO-CTS | TSV-M
# )
# args.space = "N"  # W | F | D | S | N
args.agnostic = False
args.pretrained_checkpoint = pretrained_checkpoint
assert args.space in ["W", "F", "D", "N", "S"]
################################################################################


exam_datasets = ["ag_news", "rotten_tomatoes", "cola", "sms"] 
train_datasets = exam_datasets
eval_datasets = exam_datasets
args.DATASETS = train_datasets
args.Target = range(len(train_datasets))

str_time_ = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
log = create_log_dir(args.logs_path, "log_{}_BERT.txt".format(str_time_))
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


task_vectors = [
    TaskVector(
        args.pretrained_checkpoint,
        os.path.join(args.base_dir, "checkpoints", args.model, dataset_name, "finetuned.pt"),
    ).to_cpu()
    for dataset_name in train_datasets
]

image_encoder = merging(task_vectors, log, args, args.merge, args.space, eval_datasets)

accs = []
for dataset in eval_datasets:
    metrics = eval_single_dataset(image_encoder, dataset, args)
    log.info(str(dataset) + ":" + str(metrics.get("top1") * 100) + "%")
    accs.append(metrics.get("top1") * 100)
log.info("Avg ACC:" + str(np.mean(accs)) + "%")
log.info("Time:" + str(time.time() - starttime))