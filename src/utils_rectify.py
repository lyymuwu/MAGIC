from collections import defaultdict
import copy

import torch
from src.eval import eval_single_dataset_CV, eval_single_dataset_NLP
from utils import *
import sys

sys.path.append("src/")


def get_project_component(Wts: torch.Tensor, Wm: torch.Tensor):
    """
    Project Wm ∈ R^d onto the subspace spanned by Wts ∈ R^{k x d}
    Return the projected vector Wm_proj ∈ R^d
    """
    # Wts: [k, d], Wm: [d]
    Wts = Wts  # shape [k, d]
    Wm = Wm.unsqueeze(1)  # shape [d, 1]

    # Compute projection
    A = Wts @ Wts.T  # shape [k, k]
    b = Wts @ Wm  # shape [k, 1]

    # Solve for coefficients (α): A α = b
    alpha = torch.linalg.solve(A, b)  # shape [k, 1]

    # Reconstruct projection
    Wm_proj = Wts.T @ alpha  # shape [d, 1]
    return Wm_proj.squeeze(1)  # shape [d]


def get_batched_project_component(Wts: torch.Tensor, Wm: torch.Tensor):
    """
    Project each Wm[i] ∈ R^N onto the subspace spanned by corresponding Wts[:, i, :] ∈ R^{K x N}.

    Parameters:
        Wts: shape [K, M, N], subspace basis for each sample
        Wm: shape [M, N], target vector for each sample

    Returns:
        Wm_proj: shape [M, N], projected vector for each sample
    """
    K, M, N = Wts.shape
    Wm = Wm.unsqueeze(0)  # [1, M, N]
    Wts_T = Wts.transpose(1, 2)  # [K, N, M]

    # Construct A for each sample: [M, K, K]
    Wts_trans = Wts.permute(1, 0, 2)  # [M, K, N]
    A = torch.matmul(Wts_trans, Wts_trans.transpose(1, 2))  # [M, K, K]

    # Construct b for each sample: [M, K, 1]
    b = torch.matmul(Wts_trans, Wm.permute(1, 2, 0)).squeeze(2)  # [M, K]

    # Solve α for each sample: A α = b
    alpha = torch.linalg.solve(A, b.unsqueeze(2))  # [M, K, 1]

    # Reconstruct projection: Wm_proj = Wts.T @ alpha → [M, N]
    Wts_trans_T = Wts_trans.transpose(1, 2)  # [M, N, K]
    Wm_proj = torch.matmul(Wts_trans_T, alpha).squeeze(2)  # [M, N]

    return Wm_proj


def project_to_ellipsoid(d, axes):
    """
    PyTorch implementation: Scale vector d to the ellipsoid surface formed by orthogonal vectors axes.

    Parameters:
        d: torch.Tensor, shape (N,)
        axes: list of torch.Tensor, each is an orthogonal vector with shape (N,)

    Returns:
        lambda_val: scaling factor (scalar)
        d_scaled: scaled vector Tensor
    """
    # alpha_list = [torch.dot(d, a) / torch.dot(a, a) for a in axes]
    alpha_list = [torch.dot(d, a) / torch.dot(a, a) * a.norm() for a in axes]
    norm_list = [torch.norm(a) for a in axes]
    lambda_denominator = sum(
        (alpha / norm) ** 2 for alpha, norm in zip(alpha_list, norm_list)
    )
    lambda_val = 1 / torch.sqrt(lambda_denominator)
    d_scaled = lambda_val * d
    return lambda_val, d_scaled


def project_batch_to_ellipsoid(d: torch.Tensor, axes: list[torch.Tensor]):
    """
    Batch scale d ∈ R^[M, N] to the ellipsoid surface formed by corresponding axes.

    Parameters:
        d: Tensor of shape [M, N]
        axes: list of K tensors, each of shape [M, N] (orthogonal basis for each batch)

    Returns:
        lambda_vals: Tensor of shape [M]
        d_scaled: Tensor of shape [M, N]
    """
    M, N = d.shape
    K = len(axes)

    if type(axes) is torch.Tensor:
        axes_tensor = axes.permute(1, 0, 2)  # [K, M, N] -> [M, K, N]
    else:
        axes_tensor = torch.stack(axes, dim=1)  # shape [M, K, N]
    d_expanded = d.unsqueeze(1)  # shape [M, 1, N]

    dot = (d_expanded * axes_tensor).sum(dim=2)  # shape [M, K]
    norm_sq = (axes_tensor**2).sum(dim=2)  # shape [M, K]
    norm = norm_sq.sqrt()  # shape [M, K]

    alpha = dot / norm_sq  # shape [M, K]
    term = (alpha * norm) / norm  # shape [M, K]
    lambda_denominator = (term**2).sum(dim=1)  # shape [M]
    lambda_val = 1.0 / torch.sqrt(lambda_denominator + 1e-8)  # shape [M]

    d_scaled = lambda_val.unsqueeze(1) * d  # shape [M, N]
    return lambda_val, d_scaled


def project_to_mixed_manifold(d: torch.Tensor, axes: list[torch.Tensor]):
    """
    PyTorch implementation: Scale vector d to the ellipsoid surface formed by orthogonal vectors axes.

    Parameters:
        d: torch.Tensor, shape (N,)
        axes: list of torch.Tensor, each is an orthogonal vector with shape (N,)

    Returns:
        lambda_val: scaling factor (scalar)
        d_scaled: scaled vector Tensor
    """
    d_ellipsoid = get_project_component(axes, d)
    d_circle = d - d_ellipsoid  # part projected outside the elliptic subspace
    L = axes.norm(dim=1).mean()  # calculate average norm of orthogonal vectors

    # alpha_list = [torch.dot(d_ellipsoid, a) / torch.dot(a, a) for a in axes]
    alpha_list = [torch.dot(d_ellipsoid, a) / torch.dot(a, a) * a.norm() for a in axes]
    norm_list = torch.tensor([torch.norm(a) for a in axes])
    lambda_ellipsoid = sum(
        (alpha / norm) ** 2 for alpha, norm in zip(alpha_list, norm_list)
    )

    lambda_circle = d_circle.norm().pow(2) / norm_list.mean() ** 2

    lambda_val = 1 / torch.sqrt(lambda_ellipsoid + lambda_circle)
    d_scaled = lambda_val * d
    return lambda_val, d_scaled


def optimized_project_to_mixed_manifold(d: torch.Tensor, axes: torch.Tensor):
    """
    Project d onto the manifold space of ellipsoid + hypersphere combination spanned by corresponding axes.

    Parameters:
        d: Tensor of shape [M*N]
        axes: list of K tensors, with shape [K, M*N]

    Returns:
        lambda_val: scalar
        d_scaled: Tensor of shape [M*N]
    """
    A = axes  # shape [K, N]
    A_norm_sq = (A * A).sum(dim=1)  # shape [K]
    alpha = torch.matmul(A, d) / A_norm_sq  # shape [K]

    d_ellipsoid = torch.matmul(alpha, A)  # shape [N]
    d_circle = d - d_ellipsoid  # shape [N]

    A_norm = A.norm(dim=1)  # shape [K]
    alpha_scaled = alpha * A_norm  # shape [K]
    lambda_ellipsoid = ((alpha_scaled / A_norm) ** 2).sum()

    lambda_circle = d_circle.norm().pow(2) / A_norm.mean().pow(2)

    lambda_val = 1 / torch.sqrt(lambda_ellipsoid + lambda_circle + 1e-8)
    d_scaled = lambda_val * d
    return lambda_val, d_scaled


def batch_project_to_mixed_manifold(d: torch.Tensor, axes: torch.Tensor):
    """
    Batch project each d[i] onto the manifold space of ellipsoid + hypersphere combination spanned by corresponding axes[i].

    Parameters:
        d: Tensor of shape [M, N]
        axes: list of K tensors, with shape [K, M, N]

    Returns:
        lambda_val: Tensor of shape [M]
        d_scaled: Tensor of shape [M, N]
    """
    K, M, N = axes.shape
    A = axes.permute(1, 0, 2)  # shape [M, K, N]

    A_norm_sq = (A**2).sum(dim=2)  # [M, K]
    dot = (A * d.unsqueeze(1)).sum(dim=2)  # [M, K]
    alpha = dot / (A_norm_sq + 1e-8)  # [M, K]

    d_ellipsoid = (alpha.unsqueeze(2) * A).sum(dim=1)  # [M, N]
    d_circle = d - d_ellipsoid  # [M, N]

    A_norm = A.norm(dim=2)  # [M, K]
    alpha_scaled = alpha * A_norm  # [M, K]
    lambda_ellipsoid = ((alpha_scaled / (A_norm + 1e-8)) ** 2).sum(dim=1)  # [M]

    norm_mean = A_norm.mean(dim=1)  # [M]
    lambda_circle = d_circle.norm(dim=1).pow(2) / (norm_mean**2 + 1e-8)  # [M]

    lambda_val = 1 / torch.sqrt(lambda_ellipsoid + lambda_circle + 1e-8)  # [M]
    d_scaled = lambda_val.unsqueeze(1) * d  # [M, N]
    return lambda_val, d_scaled


def batch_project_to_sphere(args, Wm: torch.Tensor, Wts: torch.Tensor):
    try:
        P = int(args.norm[1:])
    except ValueError:
        P = {"Linf": torch.inf}.get(args.norm, None)
    if P is None and args.norm != "None":
        raise NotImplementedError("Norm Not Implemented")
    
    Lp_Wts = Wts.norm(dim=-1, p=P)
    Lp_Wm = Wm.norm(dim=-1, p=P)
    ratio = Lp_Wts.mean(dim=0) / Lp_Wm if P else 1
    Wm_scaled = Wm * ratio.unsqueeze(1)
    
    return ratio, Wm_scaled

def spectral_norm_rectification(d: torch.Tensor, axes: torch.Tensor):
    """
    Align the spectral norm of each d with the average spectral norm of axes

    Parameters:
        d: Tensor of shape [M, N]
        axes: list of K tensors, with shape [K, M, N]

    Returns:
        lambda_val: Scalar
        d_scaled: Tensor of shape [M, N]
    """
    K, M, N = axes.shape
    norm_axes = torch.linalg.norm(axes, ord=2, dim=(1, 2)).mean()
    norm_d = torch.linalg.norm(d, ord=2)
    lambda_val = norm_axes / norm_d 
    d_scaled = d * lambda_val
    
    return lambda_val, d_scaled


def post_rectification(
    args,
    W_flat,
    W_merged,
    layer_name,
    global_ratio=1,
    hook_target=None,
    hook_ratios=None,
    ratios=None,
    weak_comp=None,
):
    # Weight space proj calibration is too poor and not imp here
    if args.space == "F" or args.space == "N":
        return W_merged

    try:
        P = int(args.norm[1:])
    except ValueError:
        P = {"Linf": torch.inf}.get(args.norm, None)
    if P is None and args.norm != "None":
        raise NotImplementedError("Norm Not Implemented")

    # Calculate the Manhattan Norm
    if args.space == "S" and (
        len(args.shape) == 2 and "text_projection" not in layer_name
    ):
        W_specialized = W_flat.view(W_flat.size(0), *args.shape)  # [K, W, N]
        # Calculate singular vectors for W_specialized sequentially
        S_specialized = [w_specialized.svd()[1] for w_specialized in W_specialized]
        S_specialized = torch.stack(S_specialized).mean(dim=0)
        W_merged = W_merged.reshape(args.shape)
        S_merged = W_merged.svd()[1]
        # ratio = (S_specialized / S_merged).mean() if P else 1
        ratio = (S_specialized / S_merged)[0] if P else 1
    else:
        Lp_flat = W_flat.norm(dim=-1, p=P).mean()
        Lp_merged = W_merged.norm(p=P)
        ratio = Lp_flat / Lp_merged if P else 1
        # Check if W_flat vectors are mutually orthogonal
        W_flat_normed = W_flat / W_flat.norm(dim=-1, keepdim=True)
        sim_mat = W_flat_normed @ W_flat_normed.T

        # When vectors are nearly collinear
        if sim_mat.mean() > 0.5:
            return W_merged

    # Handle weight with pool scale stability
    if weak_comp is None:
        if "c_proj" in layer_name or ".conv1" in layer_name:
            W_rectify = W_merged * ratio if ratio < 1 else W_merged
        else:
            W_rectify = W_merged * ratio if ratio > 1 else W_merged
    else:
        if layer_name in weak_comp:
            W_rectify = W_merged * ratio if ratio < 1 else W_merged
        else:
            W_rectify = W_merged * ratio if ratio > 1 else W_merged

    return W_rectify


def post_rectificationV2(args, Wts, Wm, layer_name, weak_comp=None):
    # Wts [K, M, N]
    if args.space == "W" or args.space == "D":
        ratio, W_merged_scaled = batch_project_to_mixed_manifold(Wm, Wts)
        # ratio, W_merged_scaled = batch_project_to_sphere(args, Wm, Wts)
        ratio = ratio.unsqueeze(1)
    elif args.space == "S":
        ratio, W_merged_scaled = spectral_norm_rectification(Wm, Wts)
    else:
        raise NotImplementedError("Space Not Implemented")
    print(f"Layer: {layer_name}, ratio: {ratio.mean()}")


    # Handle weight with pool scale stability
    if weak_comp is None:
        W_rectify = W_merged_scaled
    else:
        if layer_name in weak_comp:
            mask = (ratio < 1).float()
            W_rectify = Wm * (1 - mask) + (Wm * ratio) * mask
        else:
            mask = (ratio > 1).float()
            W_rectify = Wm * (1 - mask) + (Wm * ratio) * mask

    return W_rectify, ratio


def layer_wise_rectify(args, task_vectors, task_vector_target, weak_comps):
    ratios = {}
    for layer_name in task_vector_target.vector.keys():
        Wts = torch.stack([tv.vector[layer_name] for tv in task_vectors])  # [T, M, N]
        Wm = task_vector_target.vector[layer_name]  # [M, N]
        args.shape = Wts.shape[1:]

        if "conv" in layer_name:
            Wts = Wts.reshape(Wts.shape[0], Wts.shape[1], -1)
            Wm = Wm.reshape(Wm.shape[0], -1)
        elif "weight" not in layer_name or "ln" in layer_name or len(args.shape) != 2:
            ratios[layer_name] = 1
            continue

        if hasattr(args, 'target'):
            Wts = Wts[args.target:args.target+1]

        W_rectify, ratio = post_rectificationV2(
            args, Wts, Wm, layer_name, weak_comp=weak_comps
        )
        task_vector_target.vector[layer_name] = W_rectify.reshape(args.shape)
        ratios[layer_name] = ratio.mean().item()
    return ratios


def dataset_wise_rectify(
    args, task_vectors, pretrained_checkpoint, exam_datasets, encoder, 
    weak_comps, register=True, fig1=False
):
    
    registered_layer_ratios = []
    registered_layer_names = None
    image_encoder_pretrain = task_vectors[0].apply_to(
        pretrained_checkpoint, scaling_coef=0
    )

    if hasattr(args, 'target'):
        exam_datasets = exam_datasets[args.target:args.target+1]
        task_vectors = [task_vectors[args.target]]

    for target_dataset in range(len(exam_datasets)):
        task_vector_idv = task_vectors[target_dataset]
        if len(exam_datasets) == 1: # task-agnostic dataset
            estim_dataset = exam_datasets[0]
        else:
            estim_dataset = exam_datasets[target_dataset]
        encoder_idv = task_vector_idv.apply_to(
            pretrained_checkpoint, scaling_coef=1
        )

        if register:
            encoder, hook_handles = regist_hook(
                args, encoder, [image_encoder_pretrain, encoder_idv]
            )


        if estim_dataset in ["imdb", "sst2", "yelp", "ag_news", "rotten_tomatoes", "cola", "sms"]:
            eval_single_dataset_NLP(
                encoder,
                estim_dataset,
                quick_test=True,
            )
        else:        
            # Learning activation coef
            _ = eval_single_dataset_CV(
                encoder,
                estim_dataset,
                args,
                use_shuffle_test=True,
                quick_iter=True,
                constrain_batch_size=args.req, # default 1
            )


        if register:
            remove_hook(hook_handles)

        # Extract hook information
        if registered_layer_names is None:
            registered_layer_names = copy.deepcopy(hook_name_list)
        registered_layer_ratios.append(copy.deepcopy(hook_ratios_list))

        clean_list(hook_ratios_list)

    if not register:
        return None
    ############################################################################
    # Ratios register to layers
    ratios = [
        sum(layer_vals) / len(layer_vals)
        for layer_vals in zip(*registered_layer_ratios)
    ]
    dict_for_name_ratios = defaultdict(
        lambda: 1, dict(zip(registered_layer_names, ratios))
    )

    # add activation ratio for each layer
    if fig1:
        modules=[image_encoder_pretrain, encoder_idv]
    else:
        modules=[image_encoder_pretrain]
    encoder, hook_handles = regist_hook(
        args,
        encoder,
        modules=modules,
        dict_for_hook=dict_for_name_ratios,
        weak_comps=weak_comps,
    )
    
    
    # Convert all values in dict_for_name_ratios to mean().item(), then convert to dict type
    for layer_name in dict_for_name_ratios:
        if isinstance(dict_for_name_ratios[layer_name], torch.Tensor):
            dict_for_name_ratios[layer_name] = dict_for_name_ratios[layer_name].mean().item()
        else:
            dict_for_name_ratios[layer_name] = 1
    dict_for_name_ratios = dict(dict_for_name_ratios)
    # return dict_for_name_ratios
    return registered_layer_ratios
