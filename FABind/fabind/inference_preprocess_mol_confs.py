import argparse
import os
from multiprocessing import Pool

import pandas as pd
import torch
from utils.inference_mol_utils import (
    extract_torchdrug_feature_from_mol,
    generate_conformation,
    read_smiles,
)


def get_arguments():
    parser = argparse.ArgumentParser(description="Preprocess molecules.")
    parser.add_argument(
        "--index_csv",
        type=str,
        default="inference_examples/example.csv",
        help="Specify the index path for molecules.",
    )
    parser.add_argument(
        "--save_mols_dir",
        type=str,
        default="inference_examples/processed_mol",
        help="Specify where to save the processed pt.",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=10,
        help="Multiprocessing threads number",
    )
    args = parser.parse_args()
    return args


args = get_arguments()

os.system(f"mkdir -p {args.save_mols_dir}")

with open(args.index_csv, "r") as f:
    content = f.readlines()

info = []
for line in content[1:]:
    smiles, pdb, ligand_id = line.strip().split(",")
    info.append([smiles, pdb, ligand_id])

info = pd.DataFrame(info, columns=["smiles", "pdb", "ligand_id"])


def get_mol_info(idx):
    try:
        smiles = info.iloc[idx].smiles
        mol = read_smiles(smiles)
        mol = generate_conformation(mol)
        molecule_info = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)

        result = [mol, molecule_info]
        output_path = os.path.join(args.save_mols_dir, f"mol_{idx}.pt")
        torch.save(result, output_path)
    except Exception as e:
        print(
            "Failed to read molecule id ",
            idx,
            " We are skipping it. The reason is the exception: ",
            e,
        )


idx = [i for i in range(len(info))]

with Pool(processes=args.num_threads) as p:
    _ = p.map(get_mol_info, idx)

print("DONE!")
