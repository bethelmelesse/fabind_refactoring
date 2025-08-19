import numpy as np
import os

import torch

from torch_geometric.loader import DataLoader
from datetime import datetime
from utils.logging_utils import Logger
import sys
import random
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import set_seed
import shlex
import glob
import time
import pathlib

from tqdm import tqdm

from utils.fabind_inference_dataset import InferenceDataset
from utils.inference_mol_utils import write_mol
from utils.post_optim_utils import post_optimize_compound_coords
import pandas as pd
from inference_args import get_arguments

parser, args_new = get_arguments()


def Seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


command = "main_fabind.py -d 0 -m 5 --batch_size 3 --label baseline --addNoise 5 --tqdm-interval 60 --use-compound-com-cls --distmap-pred mlp --n-iter 8 --mean-layers 4 --refine refine_coord --coordinate-scale 5 --geometry-reg-step-size 0.001 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --noise-for-predicted-pocket 0.0 --clip-grad --random-n-iter --pocket-idx-no-noise --seed 128 --use-esm2-feat --pocket-pred-layers 1 --pocket-pred-n-iter 1 --center-dist-threshold 4 --pocket-cls-loss-func bce --mixed-precision no --disable-tqdm --disable-validate --log-interval 50 --optim adamw --norm-type per_sample --weight-decay 0.01 --hidden-size 512 --pocket-pred-hidden-size 128 --stage-prob 0.25"
command = shlex.split(command)

args = parser.parse_args(command[1:])
args.local_eval = args_new.local_eval
# args.eval_dir = args_new.eval_dir
args.batch_size = args_new.batch_size
args.ckpt = args_new.ckpt
args.data_path = args_new.data_path
args.resultFolder = args_new.resultFolder
args.seed = args_new.seed
args.exp_name = args_new.exp_name
args.return_hidden = args_new.return_hidden
args.confidence_task = args_new.confidence_task
args.confidence_rmsd_thr = args_new.confidence_rmsd_thr
args.confidence_thr = args_new.confidence_thr
args.test_sample_n = args_new.test_sample_n
args.disable_tqdm = False
args.tqdm_interval = 0.1
args.train_pred_pocket_noise = args_new.train_pred_pocket_noise
args.post_optim = args_new.post_optim
args.post_optim_mode = args_new.post_optim_mode
args.post_optim_epoch = args_new.post_optim_epoch
args.rigid = args_new.rigid
args.ensemble = args_new.ensemble
args.confidence = args_new.confidence
args.test_gumbel_soft = args_new.test_gumbel_soft
args.test_pocket_noise = args_new.test_pocket_noise
args.test_unseen = args_new.test_unseen
args.gs_tau = args_new.gs_tau
args.compound_coords_init_mode = args_new.compound_coords_init_mode
args.sdf_output_path_post_optim = args_new.sdf_output_path_post_optim
args.write_mol_to_file = args_new.write_mol_to_file
args.sdf_to_mol2 = args_new.sdf_to_mol2
args.n_iter = args_new.n_iter
args.redocking = args_new.redocking

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    kwargs_handlers=[ddp_kwargs], mixed_precision=args.mixed_precision
)

pre = f"{args.resultFolder}/{args.exp_name}"


os.makedirs(args.sdf_output_path_post_optim, exist_ok=True)
os.makedirs(pre, exist_ok=True)
logger = Logger(accelerator=accelerator, log_path=f"{pre}/test.log")

logger.log_message(f"{' '.join(sys.argv)}")

# torch.set_num_threads(16)
# # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
torch.multiprocessing.set_sharing_strategy("file_system")

# train, valid, test: only native pocket. train_after_warm_up, all_pocket_test include all other pockets(protein center and P2rank result)
if args.redocking:
    args.compound_coords_init_mode = "redocking"
elif args.redocking_no_rotate:
    args.redocking = True
    args.compound_coords_init_mode = "redocking_no_rotate"


def post_optim_mol(
    args,
    accelerator,
    data,
    com_coord_pred,
    com_coord_pred_per_sample_list,
    com_coord_per_sample_list,
    compound_batch,
    LAS_tmp,
    rigid=False,
):
    post_optim_device = "cpu"
    for i in range(compound_batch.max().item() + 1):
        i_mask = compound_batch == i
        com_coord_pred_i = com_coord_pred[i_mask]
        com_coord_i = data[i]["compound"].rdkit_coords

        com_coord_pred_center_i = com_coord_pred_i.mean(dim=0).reshape(1, 3)

        if rigid:
            predict_coord, loss, rmsd = post_optimize_compound_coords(
                reference_compound_coords=com_coord_i.to(post_optim_device),
                predict_compound_coords=com_coord_pred_i.to(post_optim_device),
                LAS_edge_index=None,
                mode=args.post_optim_mode,
                total_epoch=args.post_optim_epoch,
            )
            predict_coord.to(accelerator.device)
            predict_coord = (
                predict_coord
                - predict_coord.mean(dim=0).reshape(1, 3)
                + com_coord_pred_center_i
            )
            com_coord_pred[i_mask] = predict_coord
        else:
            predict_coord, loss, rmsd = post_optimize_compound_coords(
                reference_compound_coords=com_coord_i.to(post_optim_device),
                predict_compound_coords=com_coord_pred_i.to(post_optim_device),
                # LAS_edge_index=(data[i]['complex', 'LAS', 'complex'].edge_index - data[i]['complex', 'LAS', 'complex'].edge_index.min()).to(post_optim_device),
                LAS_edge_index=LAS_tmp[i].to(post_optim_device),
                mode=args.post_optim_mode,
                total_epoch=args.post_optim_epoch,
            )
            predict_coord = predict_coord.to(accelerator.device)
            predict_coord = (
                predict_coord
                - predict_coord.mean(dim=0).reshape(1, 3)
                + com_coord_pred_center_i
            )
            com_coord_pred[i_mask] = predict_coord

        com_coord_pred_per_sample_list.append(com_coord_pred[i_mask])
        com_coord_per_sample_list.append(com_coord_i)
        com_coord_offset_per_sample_list.append(data[i].coord_offset)

        mol_list.append(data[i].mol)
        uid_list.append(data[i].uid)
        smiles_list.append(data[i]["compound"].smiles)
        sdf_name_list.append(data[i].ligand_id + ".sdf")

    return


dataset = InferenceDataset(
    args_new.index_csv, args_new.pdb_file_dir, args_new.preprocess_dir
)
logger.log_message(f"data point: {len(dataset)}")
num_workers = 0
data_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    follow_batch=["x"],
    shuffle=False,
    pin_memory=False,
    num_workers=num_workers,
)

device = "cuda"
from models.model import *

model = get_model(args, logger, device)

model = accelerator.prepare(model)

model.load_state_dict(torch.load(args.ckpt))

set_seed(args.seed)

model.eval()

logger.log_message(f"Begin inference")
start_time = time.time()  # 记录开始时间

y_list = []
y_pred_list = []
com_coord_list = []
com_coord_pred_list = []
com_coord_per_sample_list = []

uid_list = []
smiles_list = []
sdf_name_list = []
mol_list = []
com_coord_pred_per_sample_list = []
com_coord_offset_per_sample_list = []

data_iter = tqdm(
    data_loader, mininterval=args.tqdm_interval, disable=not accelerator.is_main_process
)
for batch_id, data in enumerate(data_iter):
    try:
        data = data.to(device)
        LAS_tmp = []
        for i in range(len(data)):
            LAS_tmp.append(
                data[i]["compound", "LAS", "compound"].edge_index.detach().clone()
            )
        with torch.no_grad():
            com_coord_pred, compound_batch = model.inference(data)
        post_optim_mol(
            args,
            accelerator,
            data,
            com_coord_pred,
            com_coord_pred_per_sample_list,
            com_coord_per_sample_list,
            compound_batch,
            LAS_tmp=LAS_tmp,
            rigid=args.rigid,
        )
    except:
        continue

if args.sdf_to_mol2:
    from utils.sdf_to_mol2 import convert_sdf_to_mol2

if args.write_mol_to_file:
    info = pd.DataFrame(
        {"uid": uid_list, "smiles": smiles_list, "sdf_name": sdf_name_list}
    )
    info.to_csv(
        os.path.join(args.sdf_output_path_post_optim, f"uid_smiles_sdfname.csv"),
        index=False,
    )
    for i in tqdm(range(len(info))):
        save_coords = (
            com_coord_pred_per_sample_list[i] + com_coord_offset_per_sample_list[i]
        )
        sdf_output_path = os.path.join(
            args.sdf_output_path_post_optim, info.iloc[i]["sdf_name"]
        )
        mol = write_mol(
            reference_mol=mol_list[i], coords=save_coords, output_file=sdf_output_path
        )
        if args.sdf_to_mol2:
            convert_sdf_to_mol2(
                sdf_output_path, sdf_output_path.replace(".sdf", ".mol2")
            )

end_time = time.time()  # 记录开始时间
logger.log_message(f"End test, time spent: {end_time - start_time}")
