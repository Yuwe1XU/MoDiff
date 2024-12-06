import torch
import argparse
import time
from parsers.parser import Parser
from parsers.config import get_config
from trainer import Trainer_G_DT_comp as Trainer
from sampler import Sampler_G_DiT as Sampler
import pynvml
#nohup python -u main.py >> worker/nohup.out 2>&1 &


def main(work_type_args):
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    ts = time.strftime('%b%d_%H%M', time.localtime())
    args = Parser().parse()

    work_type_args.type = 'train_comp' 
    scale = '15k'
    args.config = f"Reddit/Reddit_{scale}"
    
    config = get_config(args.config, args.seed)
    config.type = args.beta_type
    
    if work_type_args.type == 'train_comp':
        config.data.file1 = f'sampled_{scale}/motif/G0_mot'
        config.data.file2 = f'sampled_{scale}/motif/G1_mot'

        trainer = Trainer(config)
        Tsf = f'{ts}_comp'
        ckpt = trainer.train(Tsf)

    elif work_type_args.type == 'eval_comp':
        ckpt = "Sep19_Re1k"
        file_name_first = f'sampled_{scale}/motif/G0_mot'
        config.ckpt = ckpt
        sampler = Sampler(config) 
        sampler.evaluation_ByCompound(ckpt, file_name_first)

    else:
        raise ValueError(f'Wrong type : {work_type_args.type}')

if __name__ == '__main__':

    work_type_parser = argparse.ArgumentParser()
    work_type_parser.add_argument('--type', type=str, default="train")
    main(work_type_parser.parse_known_args()[0])