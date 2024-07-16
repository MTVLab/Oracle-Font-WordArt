from scipy.spatial import Delaunay
import torch
import clip
import torchvision
import numpy as np
from torch.nn import functional as nnf
import torch.nn as nn
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from utils import ddim_step
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch.cuda.amp import custom_bwd, custom_fwd


class ToneLoss(nn.Module):
    def __init__(self, cfg):
        super(ToneLoss, self).__init__()
        self.dist_loss_weight = cfg.dist_loss_weight
        self.im_init = None
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(kernel_size=(cfg.pixel_dist_kernel_blur,
                                                                        cfg.pixel_dist_kernel_blur), sigma=(cfg.pixel_dist_sigma))
        self.resize = torchvision.transforms.Resize(cfg.cut_size)                                                           
       

    def set_image_init(self, im_init):
        self.im_init = im_init.permute(2, 0, 1).unsqueeze(0)
        self.im_init = self.resize(self.im_init[:,:,self.cfg.roi_box[0]:self.cfg.roi_box[1],self.cfg.roi_box[2]:self.cfg.roi_box[3]])
        self.init_blurred = self.blurrer(self.im_init)


    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1/2)*((step-300)/(20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        # print(self.init_blurred.shape, blurred_cur.shape)
        return self.mse_loss(self.init_blurred.detach(), blurred_cur) * self.get_scheduler(step)


class ConformalLoss:
    def __init__(self, parameters: EasyDict, device: torch.device, target_letter: str, shape_groups):
        self.parameters = parameters
        self.target_letter = target_letter
        self.shape_groups = shape_groups
        self.faces = self.init_faces(device)
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1) for i in range(len(self.faces))]

        with torch.no_grad():
            self.angles = []
            self.reset()

    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_
    
    def get_letter_inds(self, letter_to_insert):
        for group, l in zip(self.shape_groups, self.target_letter):
            if l == letter_to_insert:
                letter_inds = group.shape_ids
                return letter_inds[0], letter_inds[-1], len(letter_inds)

    def reset(self):
        points = torch.cat([point.clone().detach() for point in self.parameters.point])
        self.angles = self.get_angles(points)

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [self.parameters.point[i].clone().detach().cpu().numpy() for i in range(len(self.parameters.point))]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            print(c, start_ind, end_ind)
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind+1:end_ind]
            poly = Polygon(points_np[start_ind], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array([poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=np.bool)
            faces_.append(torch.from_numpy(faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_

    def __call__(self) -> torch.Tensor:
        loss_angles = 0
        points = torch.cat(self.parameters.point)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.l1_loss(angles[i], self.angles[i]))
        return loss_angles

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None


class ISMLoss(torch.nn.Module):
    def __init__(self, model_path, device, t_range=[0.02, 0.98], max_t_range=0.98):
        super(ISMLoss, self).__init__()
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
        self.unet = self.pipe.unet
        self.text_encoder = self.pipe.text_encoder
        self.tokenizer = self.pipe.tokenizer

        self.scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler", torch_dtype=torch.float16)
        self.timesteps = torch.flip(self.scheduler.timesteps, dims=(0, )).to(device)
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.scheduler.set_timesteps(self.num_train_timesteps, device=self.device)
        self.alphas = self.scheduler.alphas_cumprod.to(device) # for convenience

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.warmup_step = int(self.num_train_timesteps*(max_t_range-t_range[1]))

        self.delata_t = 35
        self.xs_delta_t = 200
        self.xs_inv_steps = 5
        self.xs_eta = 0.0
        self.denoise_guidance_scale = 1.0

        self.noise_gen = torch.Generator(self.device)
        self.noise_gen.manual_seed(256)

        self.nagative = "blurry, ugly, duplicate, poorly drawn, deformed, mosaic"
        self.embedding_inverse = self.embed_text()

    def embed_text(self, prompt=None):
        # tokenizer and embed text
        uncond_input = self.tokenizer(self.nagative, padding="max_length",
                                        max_length=self.tokenizer.model_max_length,
                                        truncation=True, return_tensors="pt")
        prompt = "" if prompt is None else prompt
        text_input = self.tokenizer(prompt, padding="max_length",
                                         max_length=uncond_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings.repeat_interleave(1, 0)
    
    def add_noise_with_cfg(self, latents, noise, 
                        ind_t, ind_prev_t, 
                        text_embeddings=None, cfg=1.0, 
                        delta_t=1, inv_steps=1,
                        is_noisy_latent=False,
                        eta=0.0):

        text_embeddings = text_embeddings.to(torch.float16)
        if cfg <= 1.0:
            uncond_text_embedding = text_embeddings.reshape(2, -1, text_embeddings.shape[-2], text_embeddings.shape[-1])[1]

        if is_noisy_latent:
            prev_noisy_lat = latents
        else:
            prev_noisy_lat = self.scheduler.add_noise(latents, noise, self.timesteps[ind_prev_t])

        cur_ind_t = ind_prev_t
        cur_noisy_lat = prev_noisy_lat

        pred_scores = []

        for i in range(inv_steps):
            # pred noise
            cur_noisy_lat_ = self.scheduler.scale_model_input(cur_noisy_lat, self.timesteps[cur_ind_t]).to(torch.float16)
            
            if cfg > 1.0:
                latent_model_input = torch.cat([cur_noisy_lat_, cur_noisy_lat_])
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)
                unet_output = self.unet(latent_model_input, timestep_model_input, 
                                encoder_hidden_states=text_embeddings).sample
                
                uncond, cond = torch.chunk(unet_output, chunks=2)
                
                unet_output = cond + cfg * (uncond - cond) # reverse cfg to enhance the distillation
            else:
                timestep_model_input = self.timesteps[cur_ind_t].reshape(1, 1).repeat(cur_noisy_lat_.shape[0], 1).reshape(-1)
                unet_output = self.unet(cur_noisy_lat_, timestep_model_input, 
                                    encoder_hidden_states=uncond_text_embedding).sample

            pred_scores.append((cur_ind_t, unet_output))

            next_ind_t = min(cur_ind_t + delta_t, ind_t)
            cur_t, next_t = self.timesteps[cur_ind_t], self.timesteps[next_ind_t]
            delta_t_ = next_t-cur_t if isinstance(self.scheduler, DDIMScheduler) else next_ind_t-cur_ind_t

            cur_noisy_lat = ddim_step(self.scheduler, unet_output, cur_t, cur_noisy_lat, -delta_t_, eta).prev_sample
            cur_ind_t = next_ind_t

            del unet_output
            torch.cuda.empty_cache()

            if cur_ind_t == ind_t:
                break

        return prev_noisy_lat, cur_noisy_lat, pred_scores[::-1]
    
    def forward(self, prompt_text, pred_rgb, grad_scale=1.0, resolution=(512, 512), step=None):
        embedding_inverse = self.embedding_inverse
        text_embeddings = self.embed_text(prompt_text)
        B = pred_rgb.shape[0]
        # K = text_embeddings.shape[0] - 1
        # 根据 as_latent 参数，编码深度图像或RGB图像为潜在空间表示。
                # encode rendered image
        x = pred_rgb * 2. - 1.
        with torch.cuda.amp.autocast():
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample())
        latents = 0.18215 * init_latent_z  # scaling_factor * init_latents
        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        # 如果噪声模板为空，则初始化噪声模板。
        # noise_temp = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, 
        #                          generator=noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        # 根据指导选项，生成噪声。
        noise = torch.randn((latents.shape[0], 4, resolution[0] // 8, resolution[1] // 8, ), dtype=latents.dtype, device=latents.device, 
                            generator=self.noise_gen) + 0.1 * torch.randn((1, 4, 1, 1), device=latents.device).repeat(latents.shape[0], 1, 1, 1)
        # nosie = [1, 4, 64, 64]
        # 调整文本嵌入的形状。TODO：文本嵌入的形状是多少？[77, 768]
        text_embeddings = text_embeddings[:, :, ...]
        text_embeddings = text_embeddings.reshape(-1, text_embeddings.shape[-2], text_embeddings.shape[-1]) # make it k+1, c * t, ...
        # 调整嵌入逆变的形状。TODO：文本嵌入的形状是多少？
        inverse_text_embeddings = embedding_inverse.unsqueeze(1).repeat(1, B, 1, 1).reshape(-1, embedding_inverse.shape[-2], embedding_inverse.shape[-1])
        # 根据预热率和指导选项，计算当前的时间步长差异。
        # TODO: warm_up_rate = 1. - min(iteration/opt.warmup_iter,1.) opt.warmup_iter=1500
        # warm_up_rate = 1. - min(step / 50, 1.)
        # current_delta_t = 35 # 取值范围[50, 100]
        current_delta_t =  self.delata_t

        # 随机选择当前时间步和前一个时间步。
        ind_t = torch.randint(self.min_step, self.max_step, (1, ), dtype=torch.long, generator=self.noise_gen, device=self.device)[0]
        ind_prev_t = max(ind_t - current_delta_t, torch.ones_like(ind_t) * 0)
        # 获取当前时间步
        t = self.timesteps[ind_t]
        # prev_t = self.timesteps[ind_prev_t]

        with torch.no_grad():
            # Step 1: sample x_s with larger steps
            starting_ind = max(ind_prev_t - self.xs_delta_t * self.xs_inv_steps, torch.ones_like(ind_t) * 0)

            _, prev_latents_noisy, pred_scores_xs = self.add_noise_with_cfg(latents, noise, ind_prev_t, starting_ind, inverse_text_embeddings, 
                                                                            self.denoise_guidance_scale, self.xs_delta_t, self.xs_inv_steps, eta=self.xs_eta)
            # Step 2: sample x_t
            _, latents_noisy, pred_scores_xt = self.add_noise_with_cfg(prev_latents_noisy, noise, ind_t, ind_prev_t, inverse_text_embeddings, 
                                                                        self.denoise_guidance_scale, current_delta_t, 1, is_noisy_latent=True)        

            pred_scores = pred_scores_xt + pred_scores_xs
            target = pred_scores[0][1]

        # 在不进行梯度计算的情况下，使用UNet模型进行前向传播，计算无条件噪声预测和文本条件噪声预测，并计算它们之间的差异。
        with torch.no_grad():
            latent_model_input = latents_noisy[None, :, ...].repeat(2, 1, 1, 1, 1).reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            tt = t.reshape(1, 1).repeat(latent_model_input.shape[0], 1).reshape(-1)

            latent_model_input = self.scheduler.scale_model_input(latent_model_input, tt[0])

            unet_output = self.unet(latent_model_input.to(torch.float16), tt.to(torch.float16), encoder_hidden_states=text_embeddings.to(torch.float16)).sample

            unet_output = unet_output.reshape(2, -1, 4, resolution[0] // 8, resolution[1] // 8, )
            noise_pred_uncond, noise_pred_text = unet_output[:1].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, ), unet_output[1:].reshape(-1, 4, resolution[0] // 8, resolution[1] // 8, )
            delta_DSD = noise_pred_text - noise_pred_uncond
        ## 根据指导比例，计算最终的噪声预测。
        pred_noise = noise_pred_uncond + 1.0 * delta_DSD

        w = lambda alphas: (((1 - alphas) / alphas) ** 0.5)
        grad = w(self.alphas[t]) * (pred_noise - target)
        grad = torch.nan_to_num(grad_scale * grad)
        loss = SpecifyGradient.apply(latents, grad)
        return loss
    
import torch.nn as nn
class SDSLoss(nn.Module):
    def __init__(self, cfg, device):
        super(SDSLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(cfg.model_path,
                                                       torch_dtype=torch.float16, variant="fp16")
        self.pipe = self.pipe.to(self.device)
        # default scheduler: PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        # beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        self.text_embeddings = None
        self.embed_text()

    def embed_text(self):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(self.cfg.prompt_text, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                         max_length=text_input.input_ids.shape[-1],
                                         return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        self.text_embeddings = self.text_embeddings.repeat_interleave(self.cfg.batch_size, 0)
        del self.pipe.tokenizer
        del self.pipe.text_encoder


    def forward(self, x_aug):
        sds_loss = 0

        # encode rendered image
        x = x_aug * 2. - 1.
        with torch.cuda.amp.autocast():
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample())
        latent_z = 0.18215 * init_latent_z  # scaling_factor * init_latents

        with torch.inference_mode():
            # sample timesteps
            timestep = torch.randint(
                low=50,
                high=min(950, 1000) - 1,  # avoid highest timestep | diffusion.timesteps=1000
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            eps = torch.randn_like(latent_z)
            # zt = alpha_t * latent_z + sigma_t * eps
            noised_latent_zt = self.pipe.scheduler.add_noise(latent_z, eps, timestep)

            # denoise
            z_in = torch.cat([noised_latent_zt] * 2)  # expand latents for classifier free guidance
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps_t_uncond, eps_t = self.pipe.unet(z_in, timestep, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)

            eps_t = eps_t_uncond + 100 * (eps_t - eps_t_uncond)

            # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
            grad_z = self.alphas[timestep]**0.5 * self.sigmas[timestep] * (eps_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

        sds_loss = grad_z.clone() * latent_z
        del grad_z

        sds_loss = sds_loss.sum(1).mean()
        return sds_loss

class ClipLoss:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.clip, _ = clip.load(cfg.clip_path, self.device, jit=False)
        self.text_source_feature = self.text_encode("A photo")
        self.image_source_feature = None

        self.text_target_feature = self.text_encode(cfg.prompt_text)

    def image_source(self, image):
        image = image.unsqueeze(0)
        image = image.permute(0, 3, 1, 2) # NHWC -> NCHW
        # image = image[:,:,105:181,270:346] flower
        # image = image[:,:,93:186,192:284] bear
        if self.cfg.roi_box:
            image = image[:,:,self.cfg.roi_box[0]:self.cfg.roi_box[1],self.cfg.roi_box[2]:self.cfg.roi_box[3]] # lighting
        self.image_source_feature = self.image_encode(image)

    def clip_normalize(self, image):
        image = nnf.interpolate(image,size=224,mode='bicubic')
        mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device)
        std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device)
        mean = mean.view(1,-1,1,1)
        std = std.view(1,-1,1,1)
        image = (image-mean)/std
        return image

    def text_encode(self, prompts):
        with torch.no_grad():
            tokens = clip.tokenize(prompts).to(self.device)
            text_features = self.clip.encode_text(tokens).detach()
            text_features = text_features.mean(axis=0, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def image_encode(self, image):
        with torch.no_grad():
            source_features = self.clip.encode_image(self.clip_normalize(image))
            source_features /= (source_features.clone().norm(dim=-1, keepdim=True))
        return source_features

    def __call__(self, x_aug):
        image_target_feature = self.clip.encode_image(self.clip_normalize(x_aug))
        image_target_feature /= (image_target_feature.clone().norm(dim=-1, keepdim=True))

        img_direction = (image_target_feature - self.image_source_feature)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        text_direction = (1 * self.text_target_feature - self.text_source_feature).repeat(image_target_feature.size(0), 1)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        loss = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1)).mean()
        return loss