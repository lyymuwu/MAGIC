from src.utils import GPU_Search, create_log_dir
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_Search())

import time
from tqdm import tqdm
from src.task_vectors import TaskVector
from src.ties_merging_utils import *
from src.eval import eval_single_dataset_CV
from src.args import parse_arguments


exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
val_datasets = ["CIFAR100"]

train_datasets = exam_datasets
eval_datasets = val_datasets[0:1]
args = parse_arguments()
str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_scaling_stability.txt'.format(str_time_))

task_vectors = [
    TaskVector(
        args.pretrained_checkpoint,
        os.path.join(args.save, dataset_name, "finetuned.pt"),
    )
    for dataset_name in train_datasets
]
task_vector_avg = sum(task_vectors) * (1.0 / len(task_vectors))


component2acc = {}
for target_w in tqdm(task_vectors[0].vector.keys()):  
    tv = copy.deepcopy(task_vector_avg)  
    for layer_name in tv.vector.keys():
        if target_w == layer_name:
            tv.vector[layer_name] *= 1.1    
        
    image_encoder = tv.apply_to(args.pretrained_checkpoint, scaling_coef=1)

    Total_ACC = 0
    for dataset in eval_datasets:
        metrics = eval_single_dataset_CV(image_encoder, dataset, args, no_print=True, quick_iter=False)
        Total_ACC += metrics['top1']
    acc = Total_ACC / len(eval_datasets)
    component2acc[target_w] = acc
    
torch.save(component2acc, 'components/component2acc_{}.pt'.format(args.model))
