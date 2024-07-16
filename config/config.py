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
    prompt_suffix: str = 'minimal flat 2d vector, with no elements being cut off.' #
    
    lr_base_point: float = 1
    lr_init: float = 0.002
    lr_final: float = 0.0008
    lr_delay_mult: float = 0.1
    lr_delay_steps: float = 100
    
    render_size: int = 600
    cut_size: int = 512
    
    use_conformal_loss: bool = True
    conformal_angeles_w: float = 0.25
    
    use_tone_loss: bool = True
    dist_loss_weight: float = 10.
    pixel_dist_kernel_blur: int = 201
    pixel_dist_sigma: int = 50
    
    level_of_cc: int = 1
    model_path: str = "/root/autodl-tmp/stable-diffusion"
    clip_path: str = "/root/autodl-tmp/clip/ViT-B-32.pt"
    prompt_text: str ='head of the lion.' + prompt_suffix,
    
    word: str = '狮'
    loss_type: str = 'clip' # ism, sds, clip
    ## 电 (185,247,253,313)
    ## 花 （105:181,270:346）
    ## 熊 （93:186,192:284）
    ## 狮  (161, 256, 355, 478)
    roi_box: tuple = (161, 256, 355, 478) #y1:y2, x1:x2
    seed: int = 0
    experiment_dir: str = f'output/{word}_seed_{str(seed)}_loss_{loss_type}'
    target: str = f"./data/init/{font}_{word}_scaled"