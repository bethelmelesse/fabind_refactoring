import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="Train your own TankBind model.")

    parser.add_argument(
        "-m",
        "--mode",
        type=int,
        default=0,
        help="mode specify the model to use.",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="0",
        help="data specify the data to use. 0 for re-docking, 1 for self-docking.",
    )
    parser.add_argument("--seed", type=int, default=42, help="seed to use.")
    parser.add_argument(
        "--gs-tau",
        type=float,
        default=1,
        help="Tau for the temperature-based softmax.",
    )
    parser.add_argument(
        "--gs-hard",
        action="store_true",
        default=False,
        help="Hard mode for gumbel softmax.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch size.")
    parser.add_argument(
        "--restart",
        type=str,
        default=None,
        help="continue the training from the model we saved from scratch.",
    )
    parser.add_argument(
        "--reload",
        type=str,
        default=None,
        help="continue the training from the model we saved.",
    )
    parser.add_argument(
        "--addNoise",
        type=str,
        default=None,
        help="shift the location of the pocket center in each training sample \
                        such that the protein pocket encloses a slightly different space.",
    )

    pair_interaction_mask = parser.add_mutually_exclusive_group()
    # use_equivalent_native_y_mask is probably a better choice.
    pair_interaction_mask.add_argument(
        "--use_y_mask",
        action="store_true",
        default=False,
        help="mask the pair interaction during pair interaction loss evaluation based on data.real_y_mask. \
                        real_y_mask=True if it's the native pocket that ligand binds to.",
    )
    pair_interaction_mask.add_argument(
        "--use_equivalent_native_y_mask",
        action="store_true",
        default=False,
        help="mask the pair interaction during pair interaction loss evaluation based on data.equivalent_native_y_mask. \
                        real_y_mask=True if most of the native interaction between ligand and protein happen inside this pocket.",
    )

    parser.add_argument(
        "--use_affinity_mask",
        type=int,
        default=0,
        help="mask affinity in loss evaluation based on data.real_affinity_mask",
    )
    parser.add_argument(
        "--affinity_loss_mode",
        type=int,
        default=1,
        help="define which affinity loss function to use.",
    )
    parser.add_argument(
        "--pred_dis",
        type=int,
        default=1,
        help="pred distance map or predict contact map.",
    )
    parser.add_argument(
        "--posweight",
        type=int,
        default=8,
        help="pos weight in pair contact loss, not useful if args.pred_dis=1",
    )
    parser.add_argument(
        "--relative_k",
        type=float,
        default=0.01,
        help="adjust the strength of the affinity loss head relative to the pair interaction loss.",
    )
    parser.add_argument(
        "-r",
        "--relative_k_mode",
        type=int,
        default=0,
        help="define how the relative_k changes over epochs",
    )
    parser.add_argument(
        "--resultFolder",
        type=str,
        default="result",
        help="information you want to keep a record.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="information you want to keep a record.",
    )
    parser.add_argument(
        "--use-whole-protein",
        action="store_true",
        default=False,
        help="currently not used.",
    )
    parser.add_argument("--data-path", type=str, default="", help="Data path.")
    parser.add_argument("--exp-name", type=str, default="", help="data path.")
    parser.add_argument(
        "--tqdm-interval",
        type=float,
        default=0.1,
        help="tqdm bar update interval",
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--pocket-coord-huber-delta", type=float, default=3.0)
    parser.add_argument(
        "--coord-loss-function",
        type=str,
        default="SmoothL1",
        choices=["MSE", "SmoothL1"],
    )
    parser.add_argument("--coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-distance-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-distance-distill-loss-weight", type=float, default=1.0)
    parser.add_argument("--pocket-cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--pocket-distance-loss-weight", type=float, default=0.05)
    parser.add_argument(
        "--pocket-cls-loss-func",
        type=str,
        default="bce",
        choices=["bce", "dice"],
    )
    # parser.add_argument("--warm-mae-thr", type=float, default=5.0)
    parser.add_argument(
        "--use-compound-com-cls",
        action="store_true",
        default=False,
        help="only use real pocket to run pocket classification task",
    )
    parser.add_argument(
        "--compound-coords-init-mode",
        type=str,
        default="pocket_center_rdkit",
        choices=[
            "pocket_center_rdkit",
            "pocket_center",
            "compound_center",
            "perturb_3A",
            "perturb_4A",
            "perturb_5A",
            "random",
            "diffdock",
        ],
    )
    parser.add_argument("--trig-layers", type=int, default=1)
    parser.add_argument(
        "--distmap-pred",
        type=str,
        default="mlp",
        choices=["mlp", "trig"],
    )
    parser.add_argument("--mean-layers", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--inter-cutoff", type=float, default=10.0)
    parser.add_argument("--intra-cutoff", type=float, default=8.0)
    parser.add_argument(
        "--refine",
        type=str,
        default="refine_coord",
        choices=["stack", "refine_coord"],
    )
    parser.add_argument("--coordinate-scale", type=float, default=5.0)
    parser.add_argument("--geometry-reg-step-size", type=float, default=0.001)
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="constant",
        choices=[
            "constant",
            "poly_decay",
            "cosine_decay",
            "cosine_decay_restart",
            "exp_decay",
        ],
    )
    parser.add_argument("--add-attn-pair-bias", action="store_true", default=False)
    parser.add_argument("--explicit-pair-embed", action="store_true", default=False)
    parser.add_argument("--opm", action="store_true", default=False)
    parser.add_argument("--add-cross-attn-layer", action="store_true", default=False)
    parser.add_argument("--rm-layernorm", action="store_true", default=False)
    parser.add_argument("--keep-trig-attn", action="store_true", default=False)
    parser.add_argument("--pocket-radius", type=float, default=20.0)
    parser.add_argument(
        "--rm-LAS-constrained-optim",
        action="store_true",
        default=False,
    )
    parser.add_argument("--rm-F-norm", action="store_true", default=False)
    parser.add_argument(
        "--norm-type",
        type=str,
        default="per_sample",
        choices=["per_sample", "4_sample", "all_sample"],
    )
    # parser.add_argument("--only-predicted-pocket-mae-thr", type=float, default=3.0)
    parser.add_argument("--noise-for-predicted-pocket", type=float, default=5.0)
    parser.add_argument("--test-random-rotation", action="store_true", default=False)
    parser.add_argument("--random-n-iter", action="store_true", default=False)
    parser.add_argument("--clip-grad", action="store_true", default=False)
    # one batch actually contains 20000 samples, not the size of training set
    parser.add_argument(
        "--sample-n",
        type=int,
        default=0,
        help="number of samples in one epoch.",
    )
    parser.add_argument("--fix-pocket", action="store_true", default=False)
    parser.add_argument("--pocket-idx-no-noise", action="store_true", default=False)
    parser.add_argument("--ablation-no-attention", action="store_true", default=False)
    parser.add_argument(
        "--ablation-no-attention-with-cross-attn",
        action="store_true",
        default=False,
    )
    parser.add_argument("--redocking", action="store_true", default=False)
    parser.add_argument("--redocking-no-rotate", action="store_true", default=False)
    parser.add_argument(
        "--pocket-pred-layers",
        type=int,
        default=1,
        help="number of layers for pocket pred model.",
    )
    parser.add_argument(
        "--pocket-pred-n-iter",
        type=int,
        default=1,
        help="number of iterations for pocket pred model.",
    )
    parser.add_argument("--use-esm2-feat", action="store_true", default=False)
    parser.add_argument("--center-dist-threshold", type=float, default=8.0)
    parser.add_argument(
        "--mixed-precision",
        type=str,
        default="no",
        choices=["no", "fp16"],
    )
    parser.add_argument("--disable-tqdm", action="store_true", default=False)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=15,
        help="used in combination with relative_k_mode.",
    )
    parser.add_argument(
        "--total-epochs",
        type=int,
        default=400,
        help="option to switch training data after certain epochs.",
    )
    parser.add_argument("--disable-validate", action="store_true", default=False)
    parser.add_argument("--disable-tensorboard", action="store_true", default=False)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--stage-prob", type=float, default=0.5)
    parser.add_argument("--pocket-pred-hidden-size", type=int, default=128)
    parser.add_argument("--local-eval", action="store_true", default=False)
    # parser.add_argument("--eval-dir", type=str, default=None)
    parser.add_argument(
        "--train-ligand-torsion-noise",
        action="store_true",
        default=False,
    )
    parser.add_argument("--train-pred-pocket-noise", type=float, default=0.0)
    parser.add_argument("--esm2-concat-raw", action="store_true", default=False)
    parser.add_argument("--test-sample-n", type=int, default=1)
    parser.add_argument("--return-hidden", action="store_true", default=False)
    parser.add_argument(
        "--confidence-task",
        type=str,
        default="classification",
        choices=["classification", "regression", "perfect"],
    )
    parser.add_argument("--confidence-rmsd-thr", type=float, default=2.0)
    parser.add_argument("--confidence-thr", type=float, default=0.5)
    parser.add_argument("--post-optim", action="store_true", default=False)
    parser.add_argument("--post-optim-mode", type=int, default=0)
    parser.add_argument("--post-optim-epoch", type=int, default=1000)
    parser.add_argument("--rigid", action="store_true", default=False)
    parser.add_argument("--ensemble", action="store_true", default=False)
    parser.add_argument("--confidence", action="store_true", default=False)
    parser.add_argument("--test-gumbel-soft", action="store_true", default=False)
    parser.add_argument("--test-pocket-noise", type=float, default=5)
    parser.add_argument("--test-unseen", action="store_true", default=False)
    parser.add_argument("--sdf-output-path-post-optim", type=str, default="")
    parser.add_argument("--write-mol-to-file", action="store_true", default=False)
    parser.add_argument("--sdf-to-mol2", action="store_true", default=False)
    parser.add_argument("--index-csv", type=str, default=None)
    parser.add_argument("--pdb-file-dir", type=str, default="")
    parser.add_argument("--preprocess-dir", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="../checkpoints/pytorch_model.bin")

    args_new = parser.parse_args()

    return parser, args_new
