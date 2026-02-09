import torch
from torch import nn
from easydict import EasyDict as edict
from einops.layers.torch import Rearrange
from einops import rearrange
import traceback
from gsplat import rasterization
import torch.nn.functional as F
import os
from transformer import TransformerBlock
from utils import (
    compute_rays, 
    compute_plucmap,
)
import numpy as np
from dpt_head import DPTHead
from torch_impl import _spherical_harmonics
from prope_custom import PropeDotProductAttention

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.RMSNorm, nn.LayerNorm)):
        module.reset_parameters()
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


class GaussianRenderer(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_c2w, test_intr, 
               W, H, sh_degree, near_plane, far_plane):
        opacity = opacity.sigmoid().squeeze(-1)
        scale = scale.exp()
        # rotation = F.normalize(rotation, p=2, dim=-1)
        test_w2c = test_c2w.float().inverse().unsqueeze(0) # (1, 4, 4)
        test_intr_i = torch.zeros(3, 3).to(test_intr.device)
        test_intr_i[0, 0] = test_intr[0]
        test_intr_i[1, 1] = test_intr[1]
        test_intr_i[0, 2] = test_intr[2]
        test_intr_i[1, 2] = test_intr[3]
        test_intr_i[2, 2] = 1
        test_intr_i = test_intr_i.unsqueeze(0) # (1, 3, 3)
        rendering, _, _ = rasterization(xyz, rotation, scale, opacity, feature,
                                        test_w2c, test_intr_i, W, H, sh_degree=sh_degree, 
                                        near_plane=near_plane, far_plane=far_plane,
                                        packed=False,
                                        absgrad=False,
                                        sparse_grad=False,                                        
                                        render_mode="RGB",
                                        backgrounds=torch.ones(1, 3).to(test_intr.device),
                                        rasterize_mode='classic') # (1, H, W, 3) 
        return rendering # (1, H, W, 3)

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
                W, H, sh_degree, near_plane, far_plane):
        ctx.save_for_backward(xyz, feature, scale, rotation, opacity, test_c2ws, test_intr)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        with torch.no_grad():
            B, V, _ = test_intr.shape
            renderings = torch.zeros(B, V, H, W, 3).to(xyz.device)
            for ib in range(B):
                for iv in range(V):
                    renderings[ib, iv:iv+1] = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib,iv], 
                                                                      test_c2ws[ib,iv], test_intr[ib,iv], W, H, sh_degree, near_plane, far_plane)
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        sh_degree = ctx.sh_degree
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            B, V, _ = test_intr.shape
            for ib in range(B):
                for iv in range(V):
                    rendering = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib,iv], 
                                                        test_c2ws[ib,iv], test_intr[ib,iv], W, H, sh_degree, near_plane, far_plane)
                    rendering.backward(grad_output[ib, iv:iv+1])

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None


class MVPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dim1 = config.model.dim1
        self.dim2 = config.model.dim2
        self.dim3 = config.model.dim3
        self.pose_keys = ["ray_o", "ray_d", "o_cross_d"]
        self.posed_image_keys = self.pose_keys + ["normalized_image"]
        self.color_dim = 3 * (self.config.model.gaussians.sh_degree + 1) ** 2
        self.opacity_dim = 1 * (self.config.model.gaussians.opacity_degree + 1) ** 2        
        self._init_tokenizers()
        self.inference_mode = hasattr(config, "inference")
        self.freeze_prev = config.model.get("freeze_prev", False)

        self.stage1 = [
            TransformerBlock(
                config.model.dim1, False, # bias
                config.model.head_dim, config.model.inter_multi,
                config.model.qk_norm)
            for _ in range(config.model.stage1_nlayer)
        ]
        self.stage1 = nn.ModuleList(self.stage1)
        self.stage2 = [
            TransformerBlock(
                config.model.dim2, False, # bias
                config.model.head_dim, config.model.inter_multi,
                config.model.qk_norm)
            for _ in range(config.model.stage2_nlayer)
        ]
        self.stage2 = nn.ModuleList(self.stage2)

        self.stage3 = [
            TransformerBlock(
                config.model.dim3, False, # bias
                config.model.head_dim, config.model.inter_multi,
                config.model.qk_norm)
            for _ in range(config.model.stage3_nlayer)
        ]
        self.stage3 = nn.ModuleList(self.stage3)
        self.apply(_init_weights)

        self.patch_size = config.model.patch_size
        self.num_register_tokens = config.model.num_register_tokens
        self.group_size = config.model.group_size

        self.register_token_init = nn.Parameter(torch.randn(1, 1, self.num_register_tokens, config.model.dim1))
        nn.init.normal_(self.register_token_init, mean=0.0, std=0.02)

        ### hard-coded Prope attention modules
        if not self.inference_mode:
            if config.training.train_stage == 1:
                self.attention2 = PropeDotProductAttention(
                head_dim=64, patches_x=30, patches_y=16,
                image_width=480, image_height=256,
                num_register_tokens=self.num_register_tokens)

                self.attention3 = PropeDotProductAttention(
                    head_dim=64, patches_x=15, patches_y=8,
                    image_width=480, image_height=256,
                    num_register_tokens=self.num_register_tokens)

            # elif config.training.train_stage in [2, 3]:
            elif config.training.train_stage == 2:
                self.attention2 = PropeDotProductAttention(
                    head_dim=64, patches_x=60, patches_y=34,
                    image_width=960, image_height=544,
                    num_register_tokens=self.num_register_tokens)

                self.attention3 = PropeDotProductAttention(
                    head_dim=64, patches_x=30, patches_y=17,
                    image_width=960, image_height=544,
                    num_register_tokens=self.num_register_tokens)

            else:
                raise NotImplementedError

        elif self.inference_mode:
            self.attention2 = PropeDotProductAttention(
                head_dim=64, patches_x=60, patches_y=34,
                image_width=960, image_height=544,
                num_register_tokens=self.num_register_tokens)

            self.attention3 = PropeDotProductAttention(
                head_dim=64, patches_x=30, patches_y=17,
                image_width=960, image_height=544,
                num_register_tokens=self.num_register_tokens)

        self.merge_block1 = nn.Conv2d(
            self.dim1, self.dim2, kernel_size=2, stride=2, 
            padding=0, bias=True, groups=self.dim1)
        self.resize_block1 = nn.Linear(self.dim1, self.dim2)

        self.merge_block2 = nn.Conv2d(
            self.dim2, self.dim3, kernel_size=2, stride=2, 
            padding=0, bias=True, groups=self.dim2)
        self.resize_block2 = nn.Linear(self.dim2, self.dim3)

        self.dpt_head = DPTHead(
            dim_in = [self.dim1, self.dim2, self.dim3],
            features = self.dim3,
            out_channels = [self.dim1, self.dim2, self.dim3],
        )

        # freeze stage1, 2
        if self.freeze_prev:
            self.register_token_init.requires_grad = False
            requires_grad(self.image_tokenizer, False)
            requires_grad(self.stage1, False)
            requires_grad(self.stage2, False)
            requires_grad(self.merge_block1, False)
            requires_grad(self.resize_block1, False)
            requires_grad(self.merge_block2, False)
            requires_grad(self.resize_block2, False)            

        if not self.inference_mode:
            from loss import LossComputer
            self.loss_computer = LossComputer(config)

    def train(self, mode=True):
        """Override the train method to keep the loss computer in eval mode"""
        super().train(mode)
        if not self.inference_mode:
            self.loss_computer.eval()

    def _init_tokenizers(self):
        """Initialize the image and target pose tokenizers, and image token decoder"""
        # Image tokenizer
        self.image_tokenizer = self._create_tokenizer(
            in_channels = self.config.model.in_channels,
            patch_size = self.config.model.patch_size,
            d_model = self.config.model.dim1
        )

        # Image token decoder (decode image tokens into pixels)
        self.gaussian_decoder = nn.Sequential(
            nn.LayerNorm(self.dim3, bias=False),
            nn.Linear(
                self.dim3,
                (self.config.model.patch_size ** 2) * \
                    (3 + self.color_dim + 3 + 4 + self.opacity_dim),
                bias=False))

    def _create_tokenizer(self, in_channels, patch_size, d_model):
        """Helper function to create a tokenizer with given config"""
        tokenizer = nn.Sequential(
            Rearrange(
                "b v c (hh ph) (ww pw) -> b (v hh ww) (ph pw c)",
                ph=patch_size, pw=patch_size),
            nn.Linear(
                in_channels * (patch_size**2), d_model, bias=False),
            nn.LayerNorm(d_model, bias=False))

        return tokenizer

    def render_one(self, xyz, feature, scale, rotation, opacity, test_c2w, test_intr, 
               W, H, sh_degree, near_plane, far_plane):
        opacity = opacity.sigmoid().squeeze(-1)
        scale = scale.exp()
        rotation = F.normalize(rotation, p=2, dim=-1)
        test_w2c = test_c2w.float().inverse().unsqueeze(0) # (1, 4, 4)
        # test_w2c = test_c2w.float().inverse()
        test_intr_i = torch.zeros(3, 3).to(test_intr.device)
        test_intr_i[0, 0] = test_intr[0]
        test_intr_i[1, 1] = test_intr[1]
        test_intr_i[0, 2] = test_intr[2]
        test_intr_i[1, 2] = test_intr[3]
        test_intr_i[2, 2] = 1
        test_intr_i = test_intr_i.unsqueeze(0) # (1, 3, 3)
        rendering, _, _ = rasterization(xyz, rotation, scale, opacity, feature,
                                        test_w2c, test_intr_i, W, H, sh_degree=sh_degree, 
                                        near_plane=near_plane, far_plane=far_plane,
                                        packed=False,
                                        absgrad=False,
                                        sparse_grad=False,                                        
                                        render_mode="RGB",
                                        backgrounds=torch.ones(1, 3).to(test_intr.device),
                                        rasterize_mode='classic') # (1, H, W, 3) 
        return rendering # (1, H, W, 3)
    
    def forward(
        self,
        input_data_dict,
        target_data_dict,
    ):
        # Do not autocast during the data processing stage
        with torch.autocast(device_type="cuda", enabled=False), torch.no_grad():
            b, v, _, h, w = input_data_dict["image"].size()
            print(f"Input image size: {b} x {v} x {h} x {w}")
            t = target_data_dict["image"].size(1)
            i_fxfycxcy = input_data_dict["fxfycxcy"]
            i_c2w = input_data_dict["c2w"]
            t_fxfycxcy = target_data_dict["fxfycxcy"]
            t_c2w = target_data_dict["c2w"]

            ray_o, ray_d = compute_plucmap(i_fxfycxcy, i_c2w, h, w)
            o_cross_d = torch.cross(ray_o, ray_d, dim=2)
            i_normalized_image = input_data_dict["image"] * 2.0 - 1.0
            i_raymap_images = torch.concat([ray_o, ray_d, o_cross_d, i_normalized_image], dim=2)

            Ks = torch.eye(3, dtype=i_c2w.dtype, device=i_c2w.device).unsqueeze(0).unsqueeze(0)
            Ks = Ks.repeat(b, v, 1, 1).clone() 
            Ks[:, :, 0, 0] = i_fxfycxcy[:, :, 0]
            Ks[:, :, 1, 1] = i_fxfycxcy[:, :, 1]
            Ks[:, :, 0, 2] = i_fxfycxcy[:, :, 2]
            Ks[:, :, 1, 2] = i_fxfycxcy[:, :, 3]
            Ks[:, :, 2, 2] = 1.0

            i_w2c = torch.inverse(i_c2w)

        context = torch.no_grad() if self.freeze_prev else nullcontext()

        with context:
            register_tokens = self.register_token_init.repeat(b, v, 1, 1)
            x = self.image_tokenizer(i_raymap_images)
            x = rearrange(x, "b (v l) d -> b v l d", v=v)
            x = torch.cat([register_tokens, x], dim=2)  # Add register tokens
            x = rearrange(x, "b v l d -> (b v) l d")
            x = self.run_stage1(x, None)
            r_tokens1, i_tokens1_prev = x[:, :self.num_register_tokens], x[:, self.num_register_tokens:]
            r_tokens1 = self.resize_block1(r_tokens1)
            hh1 = h // self.patch_size
            ww1 = w // self.patch_size
            i_tokens1 = rearrange(
                i_tokens1_prev, "b (hh ww) d -> b d hh ww",
                hh=hh1, ww=ww1)
            i_tokens1 = self.merge_block1(i_tokens1)
            i_tokens1 = rearrange(
                i_tokens1, "b d hh ww -> b (hh ww) d",
                hh=hh1//2, ww=ww1//2)
            x = torch.cat([r_tokens1, i_tokens1], dim=1)
            x = rearrange(x, "(b g v) l d -> (b g) (v l) d", g=v // self.group_size, v=self.group_size)
            info_stage2 = {
                "num_input_views": v,
                "w2c": rearrange(
                    i_w2c, "b (g v) ... -> (b g) v ...",
                    g=v // self.group_size, v=self.group_size),
                "Ks": rearrange(
                    Ks, "b (g v) ... -> (b g) v ...",
                    g=v // self.group_size, v=self.group_size),
                "attn2": self.attention2,
            }
            x = self.run_stage2(x, info_stage2)
            r_tokens2, i_tokens2_prev = x[:, :self.num_register_tokens], x[:, self.num_register_tokens:]
            r_tokens2 = self.resize_block2(r_tokens2)
            hh2 = hh1 // 2
            ww2 = ww1 // 2
            i_tokens2 = rearrange(
                i_tokens2_prev, "b (hh ww) d -> b d hh ww",
                hh=hh2, ww=ww2)
            i_tokens2 = self.merge_block2(i_tokens2)
            i_tokens2 = rearrange(
                i_tokens2, "b d hh ww -> b (hh ww) d",
                hh=hh2//2, ww=ww2//2)
            x = torch.cat([r_tokens2, i_tokens2], dim=1)
            x = rearrange(x, "(b v) l d -> b (v l) d", v=v)
        
        info_stage3 = {
            "num_input_views": v,
            "attn3": self.attention3,
            "w2c": i_w2c,
            "Ks": Ks,
        }
        x = self.run_stage3(x, info_stage3)
        i_tokens3_prev = x[:, self.num_register_tokens:]
        
        output_tokens = self.dpt_head(
            [i_tokens1_prev, i_tokens2_prev, i_tokens3_prev], [h, w], self.patch_size
        )
        output_tokens = rearrange(output_tokens, "(b v) l d -> b (v l) d", v=v)
        gaussians = self.gaussian_decoder(output_tokens)
        gaussians = rearrange(
            gaussians, "b (v hh ww) (ph pw d) -> b (v hh ph ww pw) d", v=v, 
            hh=h // self.config.model.patch_size, 
            ww=w // self.config.model.patch_size, 
            ph=self.config.model.patch_size, 
            pw=self.config.model.patch_size)
        xyz, feature, scale, rotation, opacity = torch.split(gaussians, [3, self.color_dim, 3, 4, self.opacity_dim], dim=-1)
        xyz = xyz.float() # (B, V*H*W, 3)
        feature = feature.float() # (B, V*H*W, 3 * (sh_degree + 1) ** 2)
        scale = scale.float() # (B, V*H*W, 3)
        rotation = rotation.float() # (B, V*H*W, 4)
        opacity = opacity.float() # (B, V*H*W, 1 * (opacity_degree + 1) ** 2)
        with torch.autocast(device_type="cuda", enabled=False):
            rayo_gs, rayd_gs = compute_rays(
                i_fxfycxcy, i_c2w, h, w) 
            scale = (scale + self.config.model.gaussians.scale_bias).clamp(max = self.config.model.gaussians.scale_max) 
            # opacity bias only for the sh0 component
            opacity[..., 0] = opacity[..., 0] + self.config.model.gaussians.opacity_bias
            feature = rearrange(feature, "b n (c d) -> b n d c", c=3).contiguous()
            opacity = rearrange(opacity, "b n (c d) -> b n d c", c=1).contiguous()

            dist = xyz.mean(dim=-1, keepdim=True).sigmoid() * self.config.model.gaussians.max_dist # (B, V*H*W, 1)
            xyz = dist * rayd_gs + rayo_gs

            # precompute opacity to give regularization
            if not self.inference_mode:
                dirs = xyz[:, None, :, :] - t_c2w[..., :3, 3][..., None, :] # (B, T, V*H*W, 3)
                opacity_broad = torch.broadcast_to(
                    opacity[..., None, :, :, :], (b, t, opacity.shape[1], -1, 1)
                )
                opacity_precompute = _spherical_harmonics(
                    self.config.model.gaussians.opacity_degree,
                    dirs, opacity_broad)

        if not self.inference_mode:
            gaussians = {
                "xyz": xyz,
                "feature": feature,
                "scale": scale,
                "rotation": rotation,
                "opacity": opacity_precompute,
            }

            with torch.autocast(device_type="cuda", enabled=False):
                # Rasterization
                # renderings = self.render(
                #     gaussians["xyz"], 
                #     gaussians["feature"], 
                #     gaussians["scale"], 
                #     gaussians["rotation"], 
                #     gaussians["opacity"], 
                #     t_c2w, 
                #     t_fxfycxcy, 
                #     w, h
                # )
                renderings = GaussianRenderer.apply(
                    gaussians["xyz"], 
                    gaussians["feature"], 
                    gaussians["scale"], 
                    gaussians["rotation"], 
                    gaussians["opacity"], 
                    t_c2w, 
                    t_fxfycxcy, 
                    w, h,
                    self.config.model.gaussians.sh_degree,
                    self.config.model.gaussians.near_plane,
                    self.config.model.gaussians.far_plane,
                ) # (B, V, H, W, 3)

            renderings = renderings.permute(0, 1, 4, 2, 3).contiguous() # (B, V, 3, H, W)
            
            loss_metrics = self.loss_computer(
                renderings,
                target_data_dict["image"],
            )

            with torch.autocast(device_type="cuda", enabled=False):
                ## opacity regularization
                rand_dirs = torch.randn_like(xyz)
                rand_dirs = F.normalize(rand_dirs, p=2, dim=-1) # make it a unit vector
                opacity_random = _spherical_harmonics(
                    self.config.model.gaussians.opacity_degree,
                    rand_dirs, opacity)
                opacity_random = opacity_random.sigmoid().mean()

            loss_metrics["opacity_loss"] = opacity_random * 0.001
            loss_metrics["loss"] = loss_metrics["loss"] + loss_metrics["opacity_loss"]

            result = edict(
                input=input_data_dict,
                target=target_data_dict,
                loss_metrics=loss_metrics,
                render=renderings,
                )

            return result

        else: #inference mode
            gaussians = {
                "xyz": xyz[0],
                "feature": feature[0],
                "scale": scale[0],
                "rotation": rotation[0],
                "opacity": opacity[0],
            }

            renderings = []

            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=False):
                for i in range(t):            
                    dir = gaussians["xyz"] - t_c2w[0, i:i+1, :3, 3][None, ...] # (1, N, 3)
                    opacity_i = _spherical_harmonics(
                        self.config.model.gaussians.opacity_degree,
                        dir, gaussians["opacity"][None, ...])[0] # (N, 1)
                    rendering = GaussianRenderer.render(gaussians["xyz"], gaussians["feature"], gaussians["scale"], gaussians["rotation"], opacity_i, 
                                                t_c2w[0, i], t_fxfycxcy[0, i], w, h, 
                                                self.config.model.gaussians.sh_degree, 
                                                self.config.model.gaussians.near_plane, 
                                                self.config.model.gaussians.far_plane)
                    renderings.append(rendering)
                renderings = torch.cat(renderings, dim=0)[None, ...] # (1, T, H, W, 3)
            
            renderings = renderings.permute(0, 1, 4, 2, 3).contiguous() # (B, V, 3, H, W)

            result = edict(
                input=input_data_dict,
                target=target_data_dict,
                render=renderings,
                )

            return result

    def run_stage1(self, x, info):
        for i in range(len(self.stage1)):
            x = torch.utils.checkpoint.checkpoint(
                self.stage1[i], x, False, 1, info, use_reentrant=False)
        return x
    
    def run_stage2(self, x, info):
        g = self.group_size
        v = info["num_input_views"]
        for i in range(len(self.stage2)):
            if i % 2 == 0:
                x = rearrange(
                    x, "(b g) (v l) d -> (b g v) l d", g=v//g, v=g)
                x = torch.utils.checkpoint.checkpoint(
                    self.stage2[i], x, False, 2, info, use_reentrant=False)
                x = rearrange(
                    x, "(b g v) l d -> (b g) (v l) d", g=v//g, v=g)
            else:
                x = torch.utils.checkpoint.checkpoint(
                    self.stage2[i], x, True, 2, info, use_reentrant=False)
        return rearrange(x, "(b g) (v l) d -> (b g v) l d", g=v//g, v=g)

    def run_stage3(self, x, info):
        v = info["num_input_views"]
        for i in range(len(self.stage3)):
            if i % 2 == 0:
                x = rearrange(x, "b (v l) d -> (b v) l d", v=v)
                x = torch.utils.checkpoint.checkpoint(
                    self.stage3[i], x, False, 3, info, use_reentrant=False)
                x = rearrange(x, "(b v) l d -> b (v l) d", v=v)
            else:
                x = torch.utils.checkpoint.checkpoint(
                    self.stage3[i], x, True, 3, info, use_reentrant=False)
        return rearrange(x, "b (v l) d -> (b v) l d", v=v)

    def save_input_video(self, input_intr, input_c2ws, gaussian_dict, H, W, save_path, insert_frame_num = 16):
        """
        Interpolate input frames and save rendered video
        input_intr: (V, 4), (fx, fy, cx, cy)
        input_c2ws: (V, 4, 4)
        """
        import cv2
        from camera_utils import get_interpolated_poses_many
        import subprocess
        V = input_intr.shape[0]
        device = input_intr.device
        input_intr = input_intr.detach().cpu().float()
        input_c2ws = input_c2ws.detach().cpu().float()

        input_intr_mat = torch.zeros((V, 3, 3))
        input_intr_mat[:, 0, 0] = input_intr[:, 0]
        input_intr_mat[:, 1, 1] = input_intr[:, 1]
        input_intr_mat[:, 0, 2] = input_intr[:, 2]
        input_intr_mat[:, 1, 2] = input_intr[:, 3]
        input_c2ws = torch.cat([input_c2ws, input_c2ws[:1]], dim=0) # wrap around
        input_intr_mat = torch.cat([input_intr_mat, input_intr_mat[:1]], dim=0) # wrap around
        c2ws, intr_mat, _ = get_interpolated_poses_many(input_c2ws[:, :3, :4], input_intr_mat, steps_per_transition = insert_frame_num)
        V = c2ws.shape[0]
        c2ws_mat = torch.eye(4).unsqueeze(0).repeat(V, 1, 1)
        c2ws_mat[:, :3, :4] = c2ws
        intr_fxfycxcy = torch.zeros(V, 4)
        intr_fxfycxcy[:, 0] = intr_mat[:, 0, 0]
        intr_fxfycxcy[:, 1] = intr_mat[:, 1, 1]
        intr_fxfycxcy[:, 2] = intr_mat[:, 0, 2]
        intr_fxfycxcy[:, 3] = intr_mat[:, 1, 2]
        c2ws_mat = c2ws_mat.to(device)
        intr_fxfycxcy = intr_fxfycxcy.to(device)

        xyz = gaussian_dict["xyz"].detach().float().to(device) # (N, 3)
        feature = gaussian_dict["feature"].detach().float().to(device) # (N, (sh_degree+1)**2, 3)
        scale = gaussian_dict["scale"].detach().float().to(device) # (N, 3)
        rotation = gaussian_dict["rotation"].detach().float().to(device) # (N, 4)
        opacity = gaussian_dict["opacity"].detach().float().to(device)

        renderings = []
        with torch.autocast(enabled=False, device_type="cuda"):
            for i in range(V):
                dir = xyz - c2ws_mat[i:i+1, :3, 3][None, ...] # (1, N, 3)
                opacity_i = _spherical_harmonics(
                    self.config.model.gaussians.opacity_degree,
                    dir, opacity[None, ...])[0] # (N, 1)
                rendering = self.render_one(xyz, feature, scale, rotation, opacity_i, 
                                            c2ws_mat[i], intr_fxfycxcy[i], W, H, 
                                            self.config.model.gaussians.sh_degree, 
                                            self.config.model.gaussians.near_plane, 
                                            self.config.model.gaussians.far_plane)
                rendering = rendering.squeeze(0).clamp(0, 1).cpu().numpy() # (H, W, 3)
                rendering = (rendering * 255).astype(np.uint8)
                rendering = cv2.cvtColor(rendering, cv2.COLOR_RGB2BGR)
                renderings.append(rendering)
        tmp_save_path = save_path.replace(".mp4", "_tmp.mp4")
        video_writer = cv2.VideoWriter(tmp_save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (W, H))
        for r in renderings:
            video_writer.write(r)
        video_writer.release()
        subprocess.run(f"ffmpeg -y -i {tmp_save_path} -vcodec libx264 -f mp4 {save_path} -loglevel quiet", shell=True) 
        os.remove(tmp_save_path)

    @torch.no_grad()
    def load_ckpt(self, load_path):
        if os.path.isdir(load_path):
            ckpt_names = [file_name for file_name in os.listdir(load_path) if file_name.endswith(".pt")]
            ckpt_names = sorted(ckpt_names, key=lambda x: x)
            ckpt_paths = [os.path.join(load_path, ckpt_name) for ckpt_name in ckpt_names]
        else:
            ckpt_paths = [load_path]
        try:
            checkpoint = torch.load(ckpt_paths[-1], map_location="cpu", weights_only=True)
        except:
            traceback.print_exc()
            print(f"Failed to load {ckpt_paths[-1]}")
            return None
        
        self.load_state_dict(checkpoint["ema"], strict=False)
        return 0