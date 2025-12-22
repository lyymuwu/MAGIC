from functools import partial
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

import numpy as np
import sys

sys.path.append("src/")
from src.ties_merging_utils import disjoint_merge, resolve_sign, topk_values_mask


def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)

    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]


def torch_load_old(save_path, device=None):
    with open(save_path, "rb") as f:
        classifier = pickle.load(f)
    if device is not None:
        classifier = classifier.to(device)
    return classifier


def torch_save(model, save_path):
    if os.path.dirname(save_path) != "":
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.cpu(), save_path)


def torch_load(save_path, device=None):
    model = torch.load(save_path, weights_only=False)
    if device is not None:
        model = model.to(device)
    return model


def get_logits(inputs, classifier):
    assert callable(classifier)
    if hasattr(classifier, "to"):
        classifier = classifier.to(inputs.device)
    return classifier(inputs)


def get_probs(inputs, classifier):
    if hasattr(classifier, "predict_proba"):
        probs = classifier.predict_proba(inputs.detach().cpu().numpy())
        return torch.from_numpy(probs)
    logits = get_logits(inputs, classifier)
    return logits.softmax(dim=1)


class LabelSmoothing(torch.nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def create_log_dir(path, filename="log.txt"):
    import logging

    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path + "/" + filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def remove_log(logger):
    for handler in logger.handlers:
        logger.removeHandler(handler)


def param_drop(args, W_flat):
    assert args.drop in [
        "Disjoint",
        "DARE",
        "DARE-Pure",
        "TRIM",
        "TRIM-V2",
        "None",
        "DARE-TRIM",
    ]

    if args.drop == "DARE":
        DARE_drop = args.DARE_drop / 100 if args.DARE_drop > 1 else args.DARE_drop
        mask = torch.rand_like(W_flat) > DARE_drop
        W_drop = W_flat * mask / (1 - DARE_drop)
    elif args.drop == "DARE-Pure":
        DARE_drop = args.DARE_drop / 100 if args.DARE_drop > 1 else args.DARE_drop
        mask = torch.rand_like(W_flat) > DARE_drop
        W_drop = W_flat * mask
    elif args.drop == "Disjoint":
        mask = torch.rand_like(W_flat[0]) != torch.inf
        W_drop = torch.zeros_like(W_flat)
        for idx in range(W_flat.shape[0]):
            # Get the indices of True values
            true_indices = torch.nonzero(mask).squeeze(1)
            # Select 5% of the True values to change
            num_to_change = max(1, len(mask) // W_flat.shape[0])
            indices_to_change = torch.randperm(len(true_indices))[:num_to_change]
            # Retain the selected values
            selected_indices = true_indices[indices_to_change]
            W_drop[idx][selected_indices] = W_flat[idx][selected_indices]
            # Change the selected values from True to False
            mask[selected_indices] = False
    elif args.drop == "TRIM":
        TRIM_drop = args.TRIM_drop
        K = 100 - TRIM_drop if TRIM_drop > 1 else 1 - TRIM_drop
        W_drop, *_ = topk_values_mask(W_flat, K=K, return_mask=False, flat_ptm=1)
    elif args.drop == "TRIM-V2":
        TRIM_drop = args.TRIM_drop
        K = 100 - TRIM_drop if TRIM_drop > 1 else 1 - TRIM_drop
        W_drop, *_ = topk_values_mask(W_flat, K=K, return_mask=False, flat_ptm=1)
        ratio = W_flat.norm(dim=-1, p=1).mean() / W_drop.norm(dim=-1, p=1).mean()
        W_drop = W_drop * ratio
    elif args.drop == "DARE-TRIM":
        DARE_drop = args.DARE_drop / 100 if args.DARE_drop > 1 else args.DARE_drop
        TRIM_drop = args.TRIM_drop
        K = 100 - TRIM_drop if TRIM_drop > 1 else 1 - TRIM_drop
        mask = torch.rand_like(W_flat) > DARE_drop
        W_drop = W_flat * mask / (1 - DARE_drop)
        W_drop, *_ = topk_values_mask(W_drop, K=K, return_mask=False, flat_ptm=1)
    elif args.drop == "None":
        W_drop = W_flat
    else:
        raise NotImplementedError("Drop Method Not Implemented")

    return W_drop


def param_combine(args, W_drop):
    assert args.combine in ["Disjoint", "Mean", "Sum", "SLERP"]

    if args.combine == "Disjoint":
        final_signs = resolve_sign(W_drop)
        assert final_signs is not None
        W_merge = disjoint_merge(W_drop, args.TIES_merge_func, final_signs)
    elif args.combine == "Mean":
        W_merge = W_drop.mean(dim=0)
    elif args.combine == "Sum":
        W_merge = W_drop.sum(dim=0)
    elif args.combine == "SLERP":
        W_merge = W_drop.mean(dim=0) * torch.sqrt(
            torch.tensor(2)
        )  # .pow(len(W_drop) - 1)
    else:
        raise NotImplementedError("Combine Method Not Implemented")

    return W_merge


def regist_hook(
    args,
    module,
    modules: list,
    pre_name="",
    dict_for_hook=None,
    blocklize=False,
    hook_handles=[],
    weak_comps=None,
    dbg=False,
):

    for named_children in zip(
        module.named_children(), *(m.named_children() for m in modules)
    ):
        name, child = named_children[0]
        children = [nc[1] for nc in named_children[1:]]
        full_name = pre_name + "." + name

        # Check if the current child has submodules
        is_leaf = len(list(child.children())) == 0
        if is_leaf or isinstance(child, nn.MultiheadAttention):
            # if "out_proj" in full_name:
            if dict_for_hook is not None and weak_comps is not None:
                r = dict_for_hook[full_name]
                p = [*children, args, full_name, r, weak_comps]
            elif dict_for_hook is not None:
                r = dict_for_hook[full_name]
                p = [*children, args, full_name, r]
            else:
                p = [*children, args, full_name]

            func = hook_activation
            handle = child.register_forward_hook(partial(func, params=p))
            hook_handles.append(handle)
        else:
            regist_hook(
                args,
                child,
                children,
                full_name,
                dict_for_hook,
                blocklize,
                hook_handles,
                weak_comps,
                dbg,
            )

    return module, hook_handles


hook_ratios_list = []
hook_name_list = []
magnitude_diff_list = []
magnitude_diff_calibrated_list = []
cos_sim_list = []

def hook_inference(input, output, params):
    if type(params[-1]) != list:
        *modules, args, layer_name, ratio_c = params
        weak_comp = None
    else:
        *modules, args, layer_name, ratio_c, weak_comp = params  # inference mode

    module_pre = modules[0].cuda()
    ratio_c[torch.isnan(ratio_c)] = 1

    with torch.no_grad():
        if weak_comp is not None and any(layer_name[1:] in c for c in weak_comp):
            ratio_c[ratio_c > 1] = 1

        if len(input) == 3:
            q, k, v = input
            output_pre = module_pre.forward(q, k, v)[0]
            task_feature_cur = (output[0] - output_pre) * ratio_c.unsqueeze(
                -1
            ).unsqueeze(-1)
            output = (output_pre + task_feature_cur, output[1])
        else:
            output_pre = module_pre.forward(input[0])
            task_feature_cur = output - output_pre
            task_feature_cur *= ratio_c.unsqueeze(-1).unsqueeze(-1)
            output = output_pre + task_feature_cur
    return output


def cal_task_feature(module, modules, input):
    modules = [m.cuda() for m in modules]  # Move all modules to CUDA
    module_pre, module_idv = modules
    # # Get the layer index from layer_name
    # l_idx = int(layer_name.split(".")[5]) if "resblocks" in layer_name else -1

    with torch.no_grad():
        if len(input) == 3:
            q, k, v = input
            output_pre = module_pre.forward(q, k, v)[0]
            output_idv = module_idv.forward(q, k, v)[0]
            task_feature_idv = (output_idv - output_pre).permute(1, 0, 2)
            task_feature_cur = (module.forward(q, k, v)[0] - output_pre).permute(
                1, 0, 2
            )
        else:
            output_pre = module_pre.forward(input[0])
            output_idv = module_idv.forward(input[0])
            task_feature_idv = output_idv - output_pre
            task_feature_cur = module.forward(input[0]) - output_pre

    return task_feature_idv, task_feature_cur


def hook_activation(module, input, output, params):
    import argparse
    for p in params:
        if isinstance(p, argparse.Namespace):
            args = p
            break
    # try to find fig1 in args
    if hasattr(args, 'fig1'):
        fig1 = args.fig1
    else:
        fig1 = False
        
            
    if type(params[-1]) != str and not fig1:
        output = hook_inference(input, output, params)
        return output

    if type(params[-1]) != str:
        *modules, args, layer_name, ratio, weak_comp = params
    else:
        *modules, args, layer_name = params
        ratio = None
    task_feature_idv, task_feature_cur = cal_task_feature(module, modules, input)
    if len(task_feature_idv.shape) == 3:
        task_feature_idv = task_feature_idv.permute(1, 0, 2)
        task_feature_cur = task_feature_cur.permute(1, 0, 2)
    elif len(task_feature_idv.shape) == 4:
        # If the shape is [B, C, H, W], we need to flatten it to [B, C, H*W]
        task_feature_idv = task_feature_idv.flatten(start_dim=2)
        task_feature_cur = task_feature_cur.flatten(start_dim=2)

    try:
        P = int(args.norm[1:])
    except ValueError:
        P = {"Linf": torch.inf}.get(args.norm, None)
    magnitude_idv = torch.norm(task_feature_idv, p=P, dim=-1)
    magnitude_cur = torch.norm(task_feature_cur, p=P, dim=-1)
    alpha = magnitude_idv / magnitude_cur
    print(f"Layer {layer_name}")
    print(f"Magnitude Alpha {alpha.mean().item()}")
    
    if ratio is not None:
        ratio[torch.isnan(ratio)] = 1
        if weak_comp is not None and any(layer_name[1:] in c for c in weak_comp):
            ratio[ratio > 1] = 1
        
        print("--------------------------------------------------")
        magnitude_diff = magnitude_cur - magnitude_idv
        magnitude_diff_list.append(magnitude_diff.mean().item())
        print(f"Magnitude Difference Before {magnitude_diff.mean().item()}")
        magnitude_cur_calibrated = magnitude_cur * ratio.unsqueeze(-1).unsqueeze(-1)
        magnitude_diff = magnitude_cur_calibrated - magnitude_idv
        magnitude_diff_calibrated_list.append(magnitude_diff.mean().item())
        print(f"Magnitude Difference After  {magnitude_diff.mean().item()}")
        # calculate the cosine-similarity between the two task features.
        cos_sim = F.cosine_similarity(task_feature_idv, task_feature_cur, dim=-1)
        cos_sim_list.append(cos_sim.mean().item())
        print(f"Cosine Similarity {cos_sim.mean().item()}")
        print("--------------------------------------------------")
    
    if "ln_" in layer_name:
        hook_ratios_list.append(alpha.mean())
    elif "attn" in layer_name:
        hook_ratios_list.append(alpha.mean(dim=1))
    else:
        hook_ratios_list.append(alpha.mean(dim=0)) # .mean(dim=0)
    hook_name_list.append(layer_name)

    return output


def remove_hook(hook_handles):
    for handle in hook_handles:
        handle.remove()


def clean_list(hook_ratios_list):
    while len(hook_ratios_list) > 0:
        hook_ratios_list.pop()


def GPU_Search():
    os.system("nvidia-smi -q -d Memory |grep -A5 GPU|grep Free >curtmp")
    memory_gpu = [int(x.split()[2]) for x in open("curtmp", "r").readlines()]
    os.system("rm curtmp")
    return np.argmax(memory_gpu)

def task_vector_align(task_vectors):
    for layer_name in task_vectors[0].vector.keys():
        norm_target = torch.inf
        
        for tv in task_vectors:
            Wt_target = tv.vector[layer_name]  # [M, N]
            norm_target = min(torch.norm(Wt_target, p='fro'), norm_target)
            
        for tv in task_vectors:
            Wt = tv.vector[layer_name]  # [M, N]
            norm = torch.norm(Wt, p='fro')
            tv.vector[layer_name] = Wt * (norm_target / norm)