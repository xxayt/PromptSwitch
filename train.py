import os
import random

import torch
import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup

from config.all_config import AllConfig
from datasets.data_factory import DataFactory
from model.model_factory import ModelFactory
from modules.loss import LossFactory
from trainer.trainer import Trainer
from util import get_logger
from datetime import timedelta

torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
global logger


def init_device(config, local_rank):
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    config.n_gpu = n_gpu

    if config.batch_size % config.n_gpu != 0 or config.test_batch_size % config.n_gpu != 0:
        raise ValueError("Invalid batch_size/test_batch_size and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            config.batch_size, config.n_gpu, config.test_batch_size, config.n_gpu))
    return device, n_gpu

def main():
    global logger
    config = AllConfig()
    args = config.parse_args()
    logger = get_logger(os.path.join(config.model_path, "log.txt"))

    # if config.local_rank == 0:
    #     logger.info("Effective parameters:")
    #     for key in sorted(args.__dict__):
    #         logger.info("  <<< {}: {}".format(key, args.__dict__[key]))
    device, n_gpu = init_device(config, config.local_rank)

    assert config.num_frames % config.num_prompts == 0
    assert config.num_test_frames % config.num_prompts == 0

    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    if not config.no_tensorboard:
        writer = SummaryWriter(log_dir=config.tb_log_dir)
    else:
        writer = None

    if config.seed >= 0:
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        random.seed(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_data_loader = DataFactory.get_data_loader(config, split_type='train')
    valid_data_loader  = DataFactory.get_data_loader(config, split_type='test')
    model = ModelFactory.get_model(config).to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
    #                                                   output_device=local_rank, find_unused_parameters=True)

    optimizer_grouped_params = [
        {'params': model.clip_params, 'lr': config.clip_lr},
        {'params': model.noclip_params, 'lr': config.noclip_lr}
    ]
    optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    num_training_steps = len(train_data_loader) * config.epochs
    num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    loss = LossFactory.get_loss(config)

    trainer = Trainer(model, loss, optimizer,
                      config=config,
                      device=device,
                      train_data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=scheduler,
                      writer=writer,
                      use_ema=config.use_ema,
                      logger=logger)

    trainer.train()


if __name__ == '__main__':
    main()
