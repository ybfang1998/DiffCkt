import pandas
import hydra
from omegaconf import DictConfig
import os
import pathlib

from train import train_DDM
import torch
from torch.utils.data import DataLoader, random_split
import torch.distributions as dist

import utils
from unzip_and_load import load_data, CircuitDataset

@hydra.main(version_base='1.3', config_path='./modules/configs', config_name='config')
def main(cfg: DictConfig):
    # Settings
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]

    project_name = cfg.experiments.general.project_name
    model_name = cfg.experiments.general.model_name
    experiment_type = cfg.experiments.general.experiment_type
    # Set seed
    utils.set_seed(cfg.train.seed)

    # Read Dataset
    if experiment_type == 'train':
        # Load data from unzipped files
        data_path = os.path.join(root_dir, "unzipped_data")
        circuit_dataset = load_data(dir_path=data_path)
    elif experiment_type == 'debug':
        data_path = os.path.join(root_dir, "unzipped_data/data_chunk_0.pt")
        debug_data = torch.load(data_path, weights_only=False)
        circuit_dataset = CircuitDataset(debug_data)
    elif experiment_type == 'sample':
        raise ValueError(f"Use the 'sample.py' script for sampling, not 'main.py'.")
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Data Processing
    train_loader = DataLoader(circuit_dataset, batch_size=cfg.experiments.train.batch_size, shuffle=True,
                                num_workers=cfg.train.num_workers)
    node_dist = circuit_dataset.X_dist
    # Train/Sample
    # train_node_predictor(cfg, project_name, train_loader, experiment_type)
    train_DDM(cfg, project_name, train_loader, experiment_type, node_dist)
if __name__ == '__main__':
    main()

