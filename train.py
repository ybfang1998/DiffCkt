from modules.diffusion.discrete_diffusion import DiscreteDiffusion
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning

# from modules.node_predictor import NodePredictor

import torch
import warnings
warnings.filterwarnings("ignore", category=PossibleUserWarning)
import pickle

# def train_node_predictor(cfg, project_name, train_loader, experiment_type, model_name="Node_Predictor"):
#     callbacks = []
#     if cfg.train.save_model:
#         checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{model_name}",
#                                               filename='{epoch}',
#                                               monitor='train_epoch/total_CE',
#                                               save_top_k=3,
#                                               mode='min',
#                                               every_n_epochs=1)
#         last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{model_name}", filename='last', every_n_epochs=1)
#         callbacks.append(last_ckpt_save)
#         callbacks.append(checkpoint_callback)

#     if experiment_type == 'debug':
#         print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

#     use_gpu = cfg.train.gpus > 0 and torch.cuda.is_available()

#     model = None
#     trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
#                         # strategy="find_unused_parameters_true",  # Needed to load old checkpoints
#                         accelerator='gpu' if use_gpu else 'cpu',
#                         devices=cfg.train.gpus if use_gpu else 1,
#                         max_epochs=cfg.experiments.train.epochs,
#                         # check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
#                         fast_dev_run=experiment_type == 'debug',
#                         enable_progress_bar=False,
#                         callbacks=callbacks,
#                         log_every_n_steps=50 if experiment_type != 'debug' else 1,
#                         logger=[])

#     trainer.fit(model, train_dataloaders=train_loader)

def train_DDM(cfg, project_name, train_loader, experiment_type, node_dist, model_name="Discrete_Diffusion_Model"):
    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(dirpath=f"checkpoints/{model_name}",
                                              filename='{epoch}',
                                              monitor='train_epoch/total_CE',
                                              save_top_k=3,
                                              mode='min',
                                              every_n_epochs=1)
        last_ckpt_save = ModelCheckpoint(dirpath=f"checkpoints/{model_name}", filename='last', every_n_epochs=1)
        callbacks.append(last_ckpt_save)
        callbacks.append(checkpoint_callback)

    if experiment_type == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")

    use_gpu = cfg.train.gpus > 0 and torch.cuda.is_available()

    model = DiscreteDiffusion(cfg, cfg.train.log_every_steps, node_dist)
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                        # strategy="find_unused_parameters_true",  # Needed to load old checkpoints
                        accelerator='gpu' if use_gpu else 'cpu',
                        devices=[1] if use_gpu else 1,
                        max_epochs=cfg.experiments.train.epochs,
                        # check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                        fast_dev_run=experiment_type == 'debug',
                        enable_progress_bar=False,
                        callbacks=callbacks,
                        log_every_n_steps=50 if experiment_type != 'debug' else 1,
                        logger=[])

    trainer.fit(model, train_dataloaders=train_loader)
