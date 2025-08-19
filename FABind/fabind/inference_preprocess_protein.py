import argparse
import os

import torch
from tqdm import tqdm
from utils.inference_pdb_utils import extract_esm_feature, extract_protein_structure


def get_arguments():
    parser = argparse.ArgumentParser(description="Preprocess protein.")
    parser.add_argument(
        "--pdb_file_dir",
        type=str,
        default="inference_examples/pdb_files",
        help="Specify the pdb data path.",
    )
    parser.add_argument(
        "--save_pt_dir",
        type=str,
        default="inference_examples",
        help="Specify where to save the processed pt.",
    )
    args = parser.parse_args()
    return args


args = get_arguments()

esm2_dict = {}
protein_dict = {}

data_dir = os.listdir(args.pdb_file_dir)
for pdb_file in tqdm(data_dir):
    pdb = pdb_file.split(".")[0]
    pdb_filepath = os.path.join(args.pdb_file_dir, pdb_file)
    protein_structure = extract_protein_structure(pdb_filepath)
    protein_structure["name"] = pdb
    esm2_dict[pdb] = extract_esm_feature(protein_structure)
    protein_dict[pdb] = protein_structure

result = [esm2_dict, protein_dict]
output_path = os.path.join(args.save_pt_dir, "processed_protein.pt")
torch.save(result, output_path)
