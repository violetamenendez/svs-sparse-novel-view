# Copyright 2022 BBC and University of Surrey
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import logging, coloredlogs

# torch
import torch
from pytorch_lightning import Trainer, loggers

from train import MVSNeRFSystem
from opt import config_parser

# Enable logging
logging.captureWarnings(True)
coloredlogs.install(
   level=logging.WARNING,
   fmt="%(asctime)s %(name)s:%(module)s.%(funcName)s[%(lineno)d] %(levelname)s %(message)s",
   datefmt="%F %T"
)

def test():
    torch.set_default_dtype(torch.float32)
    hparams = config_parser()
    hparams.save_dir = Path(hparams.save_dir)
    print(hparams)
    # Override training parameters
    kwargs = {}
    kwargs['configdir'] = hparams.configdir
    kwargs['datadir'] = hparams.datadir
    kwargs['expname'] = hparams.expname
    kwargs['save_dir'] = hparams.save_dir
    kwargs['dataset_name'] = hparams.dataset_name
    kwargs['finetune_scene'] = hparams.finetune_scene
    kwargs['batch_size'] = hparams.batch_size
    kwargs['patch_size'] = hparams.patch_size
    kwargs['chunk'] = hparams.chunk
    kwargs['pts_embedder'] = hparams.pts_embedder
    kwargs['depth_path'] = hparams.depth_path
    kwargs['use_closest_views'] = hparams.use_closest_views

    system = MVSNeRFSystem.load_from_checkpoint(hparams.ckpt, strict=False, **kwargs)
    print(system.hparams)

    save_dir_ckpts = hparams.save_dir / hparams.expname / 'ckpts'
    save_dir_ckpts.mkdir(parents=True, exist_ok=True)

    logger = loggers.TestTubeLogger(
        save_dir=hparams.save_dir,
        name=hparams.expname,
        debug=False,
        create_git_tag=False
    )

    hparams.num_gpus = 1
    trainer = Trainer(max_epochs=hparams.num_epochs,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      num_sanity_val_steps=0,
                      check_val_every_n_epoch = max(system.hparams.num_epochs//system.hparams.N_vis,1),
                      benchmark=True,
                      precision=system.hparams.precision)

    trainer.test(system)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    try:
        test()
    finally:
        print(torch.cuda.memory_summary(abbreviated=True))
