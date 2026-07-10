import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import os
import pathlib

import utils
import pickle

from modules.diffusion.discrete_diffusion import DiscreteDiffusion
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings("ignore", category=PossibleUserWarning)

def sample(cfg, experiment_type, model_path, predict_loader, results_path):
    use_gpu = cfg.train.gpus > 0 and torch.cuda.is_available()
    model = DiscreteDiffusion.load_from_checkpoint(model_path, weights_only=False)
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                        # strategy="find_unused_parameters_true",  # Needed to load old checkpoints
                        accelerator='gpu' if use_gpu else 'cpu',
                        devices=cfg.train.gpus if use_gpu else 1,
                        fast_dev_run=experiment_type == 'debug',
                        enable_progress_bar=False,
                        log_every_n_steps=50 if experiment_type != 'debug' else 1,
                        logger=[])
    G = trainer.predict(model, dataloaders=predict_loader)
    all_G = []
    for batch_out in G:
        if isinstance(batch_out["G"], list):
            all_G.extend(batch_out["G"])
        else:
            all_G.append(batch_out["G"])

    results = {'G': all_G}
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)

@hydra.main(version_base='1.3', config_path='./modules/configs', config_name='config')
def main(cfg: DictConfig):
    # Settings
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[0]
    project_name = cfg.experiments.general.project_name
    experiment_type = cfg.experiments.general.experiment_type
    # Set seed
    utils.set_seed(cfg.train.seed)

    # Fake predict_loader to determine the number of samples to generate
    if cfg.experiments.test.random_num_nodes:
        n_samples = cfg.experiments.test.n_samples
        indices = torch.arange(n_samples)
        predict_dataset = TensorDataset(indices)
        predict_loader = DataLoader(predict_dataset, batch_size=cfg.experiments.test.batch_size)

    model_path = to_absolute_path(cfg.experiments.test.model_path)
    results_path = to_absolute_path(cfg.experiments.test.results_path)
    sample(cfg, experiment_type, model_path, predict_loader, results_path)
if __name__ == '__main__':
    main()