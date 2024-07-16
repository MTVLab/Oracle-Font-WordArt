'''
Description: run this script to generate wordart.
Author: YuanJiang
Date: 2024-07-07 20:47:07
'''

import os
import random
from tqdm import tqdm
from losses import ConformalLoss, ISMLoss, SDSLoss, ToneLoss, ClipLoss
from easydict import EasyDict as edict
import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
import pydiffvg
import save_svg
from shapely.geometry import Polygon
from config import BaseConfig
from utils import (
    check_and_create_dir,
    get_data_augs,
    save_image,
    preprocess,
    learning_rate_decay,
    combine_word,
    create_video,)
import warnings
warnings.filterwarnings("ignore")

pydiffvg.set_print_timing(False)
gamma = 1.0

def init_shapes(svg_path):
    svg = f'{svg_path}.svg'
    canvas_width, canvas_height, shapes_init, shape_groups_init = pydiffvg.svg_to_scene(svg)
    parameters = edict()
    # path points
    parameters.point = []
    for path in shapes_init:
        path.points.requires_grad = True
        parameters.point.append(path.points)
            
    return shapes_init, shape_groups_init, parameters


if __name__ == "__main__":
    cfg = BaseConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    device = pydiffvg.get_device()

    print("preprocessing")
    preprocess(cfg.font, cfg.word, cfg.word, cfg.level_of_cc)
    if cfg.loss_type == 'ism':
        loss_fn = ISMLoss(model_path=cfg.model_path, device=device)
    elif cfg.loss_type == 'sds':
        loss_fn = SDSLoss(cfg=cfg, device=device)
    elif cfg.loss_type == 'clip':
        loss_fn = ClipLoss(cfg=cfg, device=device)
    else:
        raise ValueError("the {cfg.loss_type} is not support!")
    h, w = cfg.render_size, cfg.render_size

    data_augs = get_data_augs(cfg.cut_size)

    render = pydiffvg.RenderFunction.apply

    # initialize shape
    print('initializing shape')
    shapes, shape_groups, parameters = init_shapes(svg_path=cfg.target)

    scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
    img_init = render(w, h, 2, 2, 0, None, *scene_args)
    img_init = img_init[:, :, 3:4] * img_init[:, :, :3] + \
               torch.ones(img_init.shape[0], img_init.shape[1], 3, device=device) * (1 - img_init[:, :, 3:4])
    img_init = img_init[:, :, :3]

    print('saving init')
    filename = os.path.join(
        cfg.experiment_dir, "svg-init", "init.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(filename, w, h, shapes, shape_groups)

    num_iter = cfg.num_iter
    pg = [{'params': parameters["point"], 'lr': cfg.lr_base_point}]
    optim = torch.optim.Adam(pg, betas=(0.9, 0.9), eps=1e-6)

    if cfg.use_conformal_loss:
        # TODO:
        conformal_loss = ConformalLoss(parameters, device, cfg.word, shape_groups)
    
    if cfg.use_tone_loss:
        tone_loss = ToneLoss(cfg)
        tone_loss.set_image_init(img_init)
    
    if cfg.loss_type == 'clip':
        loss_fn.image_source(img_init)

    lr_lambda = lambda step: learning_rate_decay(step, cfg.lr_init, cfg.lr_final, num_iter,
                                                 lr_delay_steps=cfg.lr_delay_steps,
                                                 lr_delay_mult=cfg.lr_delay_mult) / cfg.lr_init

    scheduler = LambdaLR(optim, lr_lambda=lr_lambda, last_epoch=-1)  # lr.base * lrlambda_f

    print("start training")
    # training loop
    t_range = tqdm(range(num_iter))
    for step in t_range:
        optim.zero_grad()
        # render image
        scene_args = pydiffvg.RenderFunction.serialize_scene(w, h, shapes, shape_groups)
        img = render(w, h, 2, 2, step, None, *scene_args)
        
        # compose image with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=device) * (1 - img[:, :, 3:4])
        img = img[:, :, :3]

        save_image(img, os.path.join(cfg.experiment_dir, "video-png", f"iter{step:04d}.png"), gamma)
        filename = os.path.join(
            cfg.experiment_dir, "video-svg", f"iter{step:04d}.svg")
        check_and_create_dir(filename)
        save_svg.save_svg(
            filename, w, h, shapes, shape_groups)
            
        x = img.unsqueeze(0).permute(0, 3, 1, 2)  # HWC -> NCHW
        x = x.repeat(cfg.batch_size, 1, 1, 1)
        ## TODO:确定是否需要加入形变
        # x = x[:,:,105:181,270:346] flower
        if cfg.roi_box:
            x_aug = data_augs.forward(x[:,:,cfg.roi_box[0]:cfg.roi_box[1],cfg.roi_box[2]:cfg.roi_box[3]])
        else:
            x_aug = data_augs.forward(x)
        if cfg.loss_type == 'ism':
            loss = loss_fn(prompt_text=cfg.prompt_text,pred_rgb=x_aug, step=step)
        else:
            loss = loss_fn(x_aug)

        if cfg.use_conformal_loss:
            loss_angles = conformal_loss()
            loss_angles = cfg.conformal_angeles_w * loss_angles
            loss = loss + loss_angles
        if cfg.use_tone_loss:
            import torchvision
            loss = loss + tone_loss(torchvision.transforms.Resize(cfg.cut_size)(x[:,:,cfg.roi_box[0]:cfg.roi_box[1],cfg.roi_box[2]:cfg.roi_box[3]]), step)

        t_range.set_postfix({'loss': loss.item()})
        loss.backward()
        optim.step()
        scheduler.step()

    filename = os.path.join(
        cfg.experiment_dir, "output-svg", "output.svg")
    check_and_create_dir(filename)
    save_svg.save_svg(
        filename, w, h, shapes, shape_groups)

    combine_word(cfg.word, cfg.word, cfg.font, cfg.experiment_dir)

    filename = os.path.join(
        cfg.experiment_dir, "output-png", "output.png")
    check_and_create_dir(filename)
    imshow = img.detach().cpu()
    pydiffvg.imwrite(imshow, filename, gamma=gamma)

    print("saving video")
    create_video(cfg.num_iter, cfg.experiment_dir, 1)
