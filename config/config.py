'''
Description: the config file for the project
Author: YuanJiang
Date: 2024-07-07 16:37:45
'''

from dataclasses import dataclass

@dataclass
class BaseConfig:
    font: str = 'oracle'
    batch_size: int = 1
    num_iter: int = 500
    
    lr_base_point: float = 1
    lr_init: float = 0.002
    lr_final: float = 0.0008
    lr_delay_mult: float = 0.1
    lr_delay_steps: float = 100
    
    render_size: int = 600
    cut_size: int = 512
    
    use_conformal_loss: bool = False
    conformal_angeles_w: float = 3.0
    

    level_of_cc: int = 1
    model_path: str = "/root/autodl-tmp/stable-diffusion"
    prompt_text: str ='head of a dog. minimal flat 2d vector. lineal color',
    
    word: str = '狗'
    experiment_dir: str = f'output/{word}'
    target: str = f"./data/init/{font}_{word}_scaled"