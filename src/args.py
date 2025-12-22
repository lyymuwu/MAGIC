import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='/data/yayuan/.cache/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    parser.add_argument(
        "--space",
        type=str,
        default="N", 
        help="W for weight sapce, F for feature space, D for dual sapce, N for none",
    )
    parser.add_argument(
        "--merge",
        type=str,
        default="TA",
        help="Method used for model merge",
    )
    parser.add_argument(
        "--drop",
        type=str,
        default="None",
        help="Method used for drop param",
    )
    parser.add_argument(
        "--combine",
        type=str,
        default="Mean",
        help="Method used for model combine",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="L1",
        help="Norm used for model merge",
    )
    parser.add_argument(
        '--req',
        type=int,
        default=1,
        help='Required unlabeled data per class'
    )
    parser.add_argument(
        '--TRIM_drop',
        type=float,
        default=0.8,
        help='drop ratio for TIES'
    )
    parser.add_argument(
        '--DARE_drop',
        type=float,
        default=0.8,
        help='drop ratio for DARE'
    )
    parser.add_argument(
        '--Alpha',
        type=int,
        default=10,
        help='Number of magnitude-sensitivity layers'
    )
    parser.add_argument(
        '--dbg',
        type=int,
        default=-1,
        help='Debugging Mode'
    )
    parser.add_argument(
        '--dbg1',
        type=float,
        default=-1,
        help='Debugging Mode'
    )
    parser.add_argument(
        '--anchor', 
        dest='anchor', 
        action='store_true', 
        help='whether to use anchor dataset for ratio estimation'
    )
    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    ############################################################################
    args.model = "ViT-B-32" #'ViT-B-16' #'ViT-L-14'
    args.base_dir = "."
    args.data_location = os.path.join(args.base_dir, "data")
    args.save = os.path.join(args.base_dir, "checkpoints", args.model)
    args.logs_path = "logs/" + args.model
    pretrained_checkpoint = os.path.join(
        args.base_dir, "checkpoints", args.model, "zeroshot.pt"
    )
    # parsed_args.merge = (
    #     "TA"  # TIES | DARE | TA | SLERP | Consensus_TA | ISO-C | ISO-CTS | TSV-M
    # )
    # parsed_args.space = "D"  # W | F | D | S | N
    args.scaling_coef = 1
    args.agnostic = False
    args.pretrained_checkpoint = pretrained_checkpoint
    assert args.space in ["W", "F", "D", "N", "S"]
    ############################################################################
    
    if args.load is not None and len(args.load) == 1:
        args.load = args.load[0]
    return args
