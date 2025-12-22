import copy
import os
from typing import List, Optional
import torch
from tqdm import tqdm
import numpy as np
import sys

from utils_rectify import dataset_wise_rectify, layer_wise_rectify

sys.path.append("src/")
from src.ties_merging_utils import *


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


def generate_task_masks(
    tv_flat_checks: torch.Tensor,
    flat_ft: torch.Tensor,
    flat_ptm: torch.Tensor,
    tv: Optional[torch.Tensor] = None,
    tall_mask_lambda: float = 1.0,
) -> torch.Tensor:
    """
    Generate task-specific TALL masks
    TALL masks are generated as: mask_t = |theta_0 - theta_t| > |theta_mt - theta_t| * lambda

    Args:
        tv_flat_checks: individual task vectors
        flat_ft: individual theta_t (fine-tuned weights)
        flat_ptm: theta_0 (pre-trained weight)
        tv: multi-task vector
        tall_mask_lambda: hyper-parameter lambda for generating TALL masks
    Returns:
        final_mask: generated TALL masks with the given lambda, in shape (n_task, n_parameter)
    """

    print(f"Generating TALL masks.")

    if tv is None:
        tv = tv_flat_checks.sum(0)

    flat_multi = flat_ptm + tv

    original_shape = flat_ft.shape

    # generate masks by comparing the l1 distance between |theta_0 - theta_t| and |theta_mt - theta_t|
    diff_pt_ft = (flat_ptm - flat_ft).abs()
    diff_multi_ft = (flat_multi - flat_ft).abs()
    # compare the l1 distance, scaled with hyper-parameter lambda
    mask = diff_pt_ft > diff_multi_ft * tall_mask_lambda

    final_mask = (
        mask.squeeze() if original_shape == tv_flat_checks.squeeze().shape else mask
    )

    print(
        f"Average sparsity for the mask with tall_mask_lambda of {tall_mask_lambda}: {final_mask.float().mean():.4f}"
    )

    return final_mask


def construct_tall_mask(
    tv_flat_checks: torch.Tensor,
    flat_ft: torch.Tensor,
    flat_ptm: torch.Tensor,
    merged_tv: torch.Tensor,
    ptm_check: torch.Tensor,
    remove_keys: List[str],
    config,
):
    """
    Construct TALL masks for all tasks for each lambda, and store in dictionary

    Args:
        tv_flat_checks: individual task vectors
        flat_ft: individual theta_t (fine-tuned weights)
        flat_ptm: theta_0 (pre-trained weight)
        merged_tv: multi-task vector
        ptm_check: pre-trained weight as state dictionary
        remove_keys: the keys to be removed when converting between dictionary and vector
    Returns:
        tall_masks: constructed TALL masks in dictionary format of {lambda: {task: mask}}
    """
    tall_masks = {}
    for tall_mask_lambda in [0.2, 0.3, 0.4, 0.5, 0.6]:
        # generate tall masks for each lambda
        masks_at_scale = generate_task_masks(
            tv_flat_checks,
            flat_ft,
            flat_ptm,
            tall_mask_lambda=tall_mask_lambda,
            tv=merged_tv,
        )
        # convert vectors to dictionary
        masks_at_scale = [
            vector_to_state_dict(mask, ptm_check, remove_keys=remove_keys)
            for mask in masks_at_scale
        ]
        # store the masks with {dataset: mask}
        tall_masks[tall_mask_lambda] = {
            key: value for key, value in zip(config.DATASETS, masks_at_scale)
        }
    return tall_masks


def load_tall_mask(remove_keys, ptm_check):
    """Loads TALL masks from disk, unpack and transform to state dictionaries."""
    try:
        print("==== Loading TALL Masks built with Task Arithmetic ====")
        tall_masks = np.load(f"TALL_mask_{8}task.npy", allow_pickle=True).item()
    except:
        raise Exception("TALL Masks are not constructed yet.")

    # unpack masks and convert back to torch tensors
    tall_masks = {k: torch.from_numpy(np.unpackbits(v)) for k, v in tall_masks.items()}

    # convert vectors to dictionaries
    tall_masks = {
        dataset: vector_to_state_dict(mask, ptm_check, remove_keys=remove_keys)
        for dataset, mask in tall_masks.items()
    }

    return tall_masks


def construct_consensus_mask(tall_masks, ptm_check, prun_thre_k, remove_keys=[]):
    """
    Generate consensus mask by filtering out least-used parameters

    Args:
        ptm_check: pretrained_checkpoint as state dictionary
        prun_thre_k: weight-pruning threhold, stands for the least number of activated tasks for a parameter to be preserved from pruning
                if prun_thre_k is set to 2: remove both catastrophic and selfish weights;
                if prun_thre_k is set to 1: remove only catastrophic weights;
                if prun_thre_k is set to 0: remove no weights -> reduce to TA or TIES
                if prun_thre_k is set to > num_tasks: remove all weights -> reduce to zero-shot
    Returns:
        consensus_mask_vector: constructed consensus mask as vector (boolean in shape (n_parameter, ))
    """

    print("==== Generating Consensus Mask ====")
    # load TALL masks (in shape (n_task, n_parameter))
    tall_masks = load_tall_mask(remove_keys, ptm_check)
    tall_masks = list(tall_masks.values())
    # #########################
    # # unpack masks and convert back to torch tensors
    # tall_masks = {k: torch.from_numpy(np.unpackbits(v)) for k, v in tall_masks.items()}
    # # convert vectors to dictionaries
    # tall_masks = {
    #     dataset: vector_to_state_dict(mask, ptm_check, remove_keys=remove_keys) for dataset, mask in tall_masks.items()
    # }
    # #########################
    # tall_masks = tall_masks[0.4] # 0.2-0.6
    # tall_masks = list(tall_masks.values())

    # generate consensus masks
    consensus_mask = copy.deepcopy(tall_masks[0])
    for key, value in consensus_mask.items():
        consensus_mask[key] = torch.zeros_like(value)
        # count for each parameter, the tasks it has been activated for
        for mask in tall_masks:
            consensus_mask[key] = consensus_mask[key] + mask[key].float()
        # filter out the least-activated parameters based on given threshold
        consensus_mask[key] = consensus_mask[key].float() >= prun_thre_k
    consensus_mask_vector = state_dict_to_vector(
        consensus_mask, remove_keys=remove_keys
    )

    return consensus_mask_vector


def mask_input_with_mask_rate(
    input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str
):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert (
        0.0 <= mask_rate <= 1.0
    ), f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(
            torch.full_like(input=input_tensor, fill_value=mask_rate)
        ).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert (
            mask_strategy == "magnitude"
        ), f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(
            k=num_mask_params, dim=0, keepdim=True
        )
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor


def WA(task_vector_avg, task_vectors, config):
    config.scaling_coef = 1
    return task_vector_avg


def TA(task_vector_avg, task_vectors, config):
    config.scaling_coef = 0.3
    return task_vector_avg * len(task_vectors)


def TIES(task_vector_avg, task_vectors, args):
    ft_checks = [
        torch.load(
            os.path.join(
                args.base_dir,
                "checkpoints",
                args.model,
                dataset_name,
                "finetuned.pt",
            ),
            weights_only=False,
        ).state_dict()
        for dataset_name in args.DATASETS
    ]
    ptm_check = torch.load(
        args.pretrained_checkpoint, weights_only=False
    ).state_dict()
    check_parameterNamesMatch(ft_checks + [ptm_check])

    remove_keys = []
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    tv_flat_checks = flat_ft - flat_ptm
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check
    )
    assert all(
        [
            check_state_dicts_equal(
                vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]
            )
            for i in range(len(ft_checks))
        ]
    )

    K = 20
    merge_func = "dis-sum"
    args.scaling_coef_ = 0.3

    merged_tv = ties_merging(
        tv_flat_checks,
        reset_thresh=K,
        merge_func=merge_func,
    )
    merged_state_dict = vector_to_state_dict(
        merged_tv, ptm_check, remove_keys=remove_keys
    )
    task_vector_avg.vector = merged_state_dict
    return task_vector_avg

def layer_wise_TIES(task_vector_avg, task_vectors, config):
    """
    Layer-wise TIES merging function.
    This function merges task vectors layer by layer using TIES method.
    """
    for key in task_vectors[0].vector:
        Wts = torch.stack([tv.vector[key] for tv in task_vectors]) # [T, M, N]
        W_flat = Wts.reshape(len(Wts), -1)

        # Apply TIES merging on the current layer
        merged_layer = ties_merging(
            W_flat,
            reset_thresh=20,  # K value
            merge_func="dis-mean",  # or "dis-sum"
        )
        # Reshape back to original layer shape
        task_vector_avg.vector[key] = merged_layer.reshape(Wts.shape[1:])  

    return task_vector_avg


def DARE(task_vector_avg, task_vectors, config):
    with torch.no_grad():
        drop_p = 0.9
        for key in task_vectors[0].vector:
            Wts = [task_vector.vector[key] for task_vector in task_vectors]
            Wts = torch.stack(Wts)

            Wts = mask_input_with_mask_rate(Wts, 1 - drop_p, True, "random")
            Wm = Wts.mean(dim=0)
            task_vector_avg.vector[key] = Wm

    return task_vector_avg


def SLERP(task_vector_avg, task_vectors, config):
    """
    SLERP: Spherical Linear Interpolation
    This method is used to merge task vectors by interpolating them on the unit sphere.
    """
    assert len(task_vectors) == 2, "SLERP requires two task vectors to merge."
    global_tmp = []
    with torch.no_grad():
        for key in task_vectors[0].vector:
            Wts = [task_vector.vector[key] for task_vector in task_vectors]
            Wts = torch.stack(Wts)

            v0, v1 = Wts[0], Wts[1]
            v0_norm = v0 / v0.norm(dim=-1, keepdim=True)
            v1_norm = v1 / v1.norm(dim=-1, keepdim=True)

            dot = (v0_norm * v1_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
            omega = torch.acos(dot)  # angle between two vectors
            sin_omega = torch.sin(omega)

            # Handle boundary case when sin(omega) is 0 (when angle is very small, approximate to linear interpolation)
            near_zero = sin_omega.abs() < 1e-6

            t = 0.5  # interpolation coefficient, 0.5 means midpoint between two vectors
            slerp_vec = (
                torch.sin((1 - t) * omega) / sin_omega * v0
                + torch.sin(t * omega) / sin_omega * v1
            )
            lerp_vec = (1 - t) * v0 + t * v1  # fallback to linear

            Wm = torch.where(near_zero, lerp_vec, slerp_vec)
            task_vector_avg.vector[key] = Wm
            tmp = torch.sin(t * omega) / sin_omega + torch.sin((1 - t) * omega) / sin_omega
            if not tmp.isnan().any():
                global_tmp.append(tmp.mean().item())
        print(np.mean(global_tmp))
    return task_vector_avg


# FIXME: This method seems to depend on different checkpoints
def Consensus_TA(task_vector_avg, task_vectors, config):
    config.K = 20

    ft_checks = [
        torch.load(
            os.path.join(
                config.base_dir,
                "checkpoints",
                config.model,
                dataset_name,
                "finetuned.pt",
            ),
            weights_only=False,
        ).state_dict()
        for dataset_name in config.DATASETS
    ]
    ptm_check = torch.load(
        config.pretrained_checkpoint, weights_only=False
    ).state_dict()
    check_parameterNamesMatch(ft_checks + [ptm_check])

    remove_keys = []
    print(f"Flattening out Checkpoints")
    flat_ft = torch.vstack(
        [state_dict_to_vector(check, remove_keys) for check in ft_checks]
    )
    flat_ptm = state_dict_to_vector(ptm_check, remove_keys)

    tv_flat_checks = flat_ft - flat_ptm
    assert check_state_dicts_equal(
        vector_to_state_dict(flat_ptm, ptm_check, remove_keys), ptm_check
    )
    assert all(
        [
            check_state_dicts_equal(
                vector_to_state_dict(flat_ft[i], ptm_check, remove_keys), ft_checks[i]
            )
            for i in range(len(ft_checks))
        ]
    )

    print(f"Using Task Arithmetic for constructing multi-task vector")
    tv_flat_checks, _ = topk_values_mask(tv_flat_checks, K=config.K, return_mask=False)
    merged_tv = tv_flat_checks.sum(dim=0)

    eval_masks = None
    consensus_mask = construct_consensus_mask(eval_masks, ptm_check, 2, remove_keys)

    tv_flat_checks, _ = topk_values_mask(
        tv_flat_checks, K=config.K, return_mask=False
    )  # top-k mag filtering
    merged_tv = tv_flat_checks.sum(dim=0)
    # apply the consensus mask to filter multi-task vector
    merged_tv = merged_tv * consensus_mask

    merged_state_dict = vector_to_state_dict(
        merged_tv, ptm_check, remove_keys=remove_keys
    )
    task_vector_avg.vector = merged_state_dict
    return task_vector_avg


def TSVM(task_vector_avg, task_vectors, args):  # task_vector_sum.vector, task_vectors
    print("Weight Decomposition...")
    for key, value in tqdm(task_vector_avg.vector.items()):
        if len(task_vectors[0].vector[key].shape) == 2 and "text_projection" not in key:
            pass
        else:
            continue

        if args.model == "BERT":
            if "output.dense.weight" not in key:
                continue
        else:
            if (
                "token_embedding" in key
                or "positional_embedding" in key
                or "embeddings" in key
            ):
                continue

        device = task_vectors[0].vector[key].device
        sv_reduction = 1 / len(task_vectors)
        for i, task_vector in enumerate(task_vectors):
            vec = task_vector.vector[key].cuda()
            u, s, v = torch.linalg.svd(vec, full_matrices=False)

            if i == 0:
                print(f"Computed SVD for {key}...")
                sum_u = torch.zeros_like(u, device="cuda")
                sum_s = torch.zeros_like(s, device="cuda")
                sum_v = torch.zeros_like(v, device="cuda")
            reduced_index_s = int(s.shape[0] * sv_reduction)

            # select only the first reduced_index_s columns of u and place them
            cur_len = sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s].shape[1]
            sum_u[:, i * reduced_index_s : (i + 1) * reduced_index_s] = u[:, :cur_len]
            sum_s[i * reduced_index_s : (i + 1) * reduced_index_s] = s[:cur_len]
            # select only the first reduced_index_s rows of v and place them
            sum_v[i * reduced_index_s : (i + 1) * reduced_index_s, :] = v[:cur_len, :]

        try:
            u_u, s_u, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, s_v, v_v = torch.linalg.svd(sum_v, full_matrices=False)
        except:
            noise = torch.eye(sum_u.shape[0], sum_u.shape[1]) * 1e-10
            u_u, s_u, v_u = torch.linalg.svd(sum_u + noise, full_matrices=False)
            noise = torch.eye(sum_v.shape[0], sum_v.shape[1]) * 1e-10
            u_v, s_v, v_v = torch.linalg.svd(sum_v + noise, full_matrices=False)

        U_tsm, V_tsm, S_tsm = u_u @ v_u, u_v @ v_v, sum_s
        W_TSM = U_tsm @ torch.diag(S_tsm) @ V_tsm
        task_vector_avg.vector[key] = W_TSM.to(device)

    return task_vector_avg


def iso_c(task_vector_avg, task_vectors, config):
    print("Computing SVD...")
    with torch.no_grad():
        for key in task_vectors[0].vector:
            Wts = [task_vector.vector[key] for task_vector in task_vectors]
            task_vector_avg.vector[key] = sum(Wts) / len(Wts)
            device = task_vector_avg.vector[key].device

            if (
                len(task_vectors[0].vector[key].shape) == 2
                and "text_projection" not in key
            ):
                Wm = (task_vector_avg.vector[key] * len(Wts)).cuda()
                Wts = torch.stack(Wts).cuda()
                U, S, V = torch.linalg.svd(Wm, full_matrices=False)
                S_mean = torch.ones_like(S) * S.mean()

                W = torch.linalg.multi_dot(
                    (
                        U,
                        torch.diag(S_mean),
                        V,
                    )
                )
                task_vector_avg.vector[key] = W.to(device)

    return task_vector_avg


@torch.no_grad()
def iso_cts(task_vector_avg, task_vectors, config):
    common_space_fraction = 0.8
    print("Computing SVD...")
    for key in task_vectors[0].vector:
        shape_ = task_vectors[0].vector[key].shape

        is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
        if not is_2d_matrix:
            print(f"Combining by avg {key}...")
            for i, (task_vector, dataset) in enumerate(
                zip(task_vectors, config.DATASETS)
            ):
                vec = task_vector.vector[key]
                if i == 0:
                    task_vector_avg.vector[key] = vec.clone()
                else:
                    task_vector_avg.vector[key] += (
                        vec - task_vector_avg.vector[key]
                    ) / (i + 1)
            continue

        print(f"Computing common space using sum for {key}...")
        combined_w = sum([task_vector.vector[key] for task_vector in task_vectors])

        ### Calculate the common space size (making sure that task specific space is equally divisible) ###
        common_space_index_s = int(min(shape_) * common_space_fraction)
        _task_specific_total_space_index_s = round(
            (min(shape_) - common_space_index_s) / len(config.DATASETS)
        ) * len(config.DATASETS)
        common_space_index_s = min(shape_) - _task_specific_total_space_index_s

        u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
        common_space_u = u[:, :common_space_index_s]
        common_space_s = s[:common_space_index_s]
        common_space_v = v[:common_space_index_s, :]
        ###################################################################

        ### Calculate task specific space ###
        n_dims_per_task = int(
            (min(shape_) - common_space_index_s) / len(config.DATASETS)
        )
        for i, task_vector in enumerate(task_vectors):
            w = task_vector.vector[key]

            # calculate the projection onto task specific space to remove the common space
            w_ts = w - common_space_u @ common_space_u.T @ w
            u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

            if i == 0:
                combined_space_u = torch.zeros_like(u_ts)
                combined_space_s = torch.zeros_like(s_ts)
                combined_space_v = torch.zeros_like(v_ts)

            combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[
                :, :n_dims_per_task
            ]
            combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[
                :n_dims_per_task
            ]
            combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[
                :n_dims_per_task, :
            ]
        ###################################################################

        combined_space_u[
            :,
            len(config.DATASETS)
            * n_dims_per_task : len(config.DATASETS)
            * n_dims_per_task
            + common_space_index_s,
        ] = common_space_u
        combined_space_s[
            len(config.DATASETS)
            * n_dims_per_task : len(config.DATASETS)
            * n_dims_per_task
            + common_space_index_s
        ] = common_space_s
        combined_space_v[
            len(config.DATASETS)
            * n_dims_per_task : len(config.DATASETS)
            * n_dims_per_task
            + common_space_index_s,
            :,
        ] = common_space_v

        ### Orthogonalize combined_space_u and combined_space_v ###
        try:
            u_combined_space_u, s_combined_space_u, v_combined_space_u = (
                torch.linalg.svd(combined_space_u, full_matrices=False)
            )
            u_combined_space_v, s_combined_space_v, v_combined_space_v = (
                torch.linalg.svd(combined_space_v, full_matrices=False)
            )
        except:
            noise = (
                torch.eye(combined_space_u.shape[0], combined_space_u.shape[1]) * 1e-8
            )
            u_combined_space_u, s_combined_space_u, v_combined_space_u = (
                torch.linalg.svd(combined_space_u + noise, full_matrices=False)
            )
            noise = (
                torch.eye(combined_space_v.shape[0], combined_space_v.shape[1]) * 1e-8
            )
            u_combined_space_v, s_combined_space_v, v_combined_space_v = (
                torch.linalg.svd(combined_space_v + noise, full_matrices=False)
            )
        combined_space_u = u_combined_space_u @ v_combined_space_u
        combined_space_v = u_combined_space_v @ v_combined_space_v
        ###################################################################

        combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()

        task_vector_avg.vector[key] = torch.linalg.multi_dot(
            (
                combined_space_u,
                torch.diag(combined_space_s),
                combined_space_v,
            )
        )

    return task_vector_avg




def merging(task_vectors, log, args, merge_func, calibration_space="N", val_datasets=[]):
    assert calibration_space in ["W", "F", "D", "N"], "Wrong calibration space setting!"

    if calibration_space != "N":
        Alpha = args.Alpha   

    if calibration_space == "F":
        assert len(val_datasets) > 0, "Validation datasets should be provided for Feature Space Calibration!"


    args.merge = merge_func
    model = args.model
    pretrained_checkpoint = args.pretrained_checkpoint
         


    task_vector_avg = copy.deepcopy(sum(task_vectors)) * (1 / len(task_vectors))

    # Merging methods
    merge_methods = {
        "WA": WA,
        "TA": TA,
        "TIES": layer_wise_TIES,
        "DARE": DARE,
        "SLERP": SLERP,
        "Consensus_TA": Consensus_TA,
        "TSV-M": TSVM,
        "Iso-C": iso_c,
        "Iso-CTS": iso_cts,
    }
    merge_methods[merge_func](task_vector_avg, task_vectors, args)


    ######################### Weak Component Selection #########################
    try:
        component2acc = torch.load(
            f"components/component2acc_{model}.pt", weights_only=False
        )
        sorted_component2acc = sorted(component2acc.items(), key=lambda x: x[1])
        weak_comps = [comp for comp, acc in sorted_component2acc[:Alpha]]
    except:
        weak_comps = None
    

    ###################### Weight Space Calibration (WSC) ######################
    if calibration_space in ["W", "D"]:
        args.scaling_coef = 1
        layer_wise_rectify(args, task_vectors, task_vector_avg, weak_comps)


    image_encoder = task_vector_avg.apply_to(
        pretrained_checkpoint, scaling_coef=args.scaling_coef
    )
    log.info("*" * 20 + "Merge Method:" + str(args.merge) + "*" * 20)


    ###################### Feature Space Calibration (FSC) ######################
    if calibration_space in ["F", "D"]:
        dataset_wise_rectify(
            args, task_vectors, pretrained_checkpoint,
            val_datasets, image_encoder, weak_comps,
        )
    
    return image_encoder