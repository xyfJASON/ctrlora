import einops

import torch
import torch.nn as nn
from torchvision.utils import make_grid

from cldm.ddim_hacked import DDIMSampler
from cldm.cldm import ControlNet, ControlLDM
from cldm.lora import LoRALinearLayer, LoRACompatibleLinear
from ldm.util import log_txt_as_img
from ldm.modules.diffusionmodules.util import timestep_embedding


class ControlNetPretrain(ControlNet):
    def __init__(self, lora_rank, tasks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_rank = lora_rank
        self.tasks = tasks
        self.n_tasks = len(tasks)

        # delete input hint block
        del self.input_hint_block

        # define lora layers for each task
        self.loras_dict = nn.ModuleDict({})
        linear_modules = [m for n, m in self.named_modules() if isinstance(m, nn.Linear)]
        for task in self.tasks:
            loras = nn.ModuleList([])
            for m in linear_modules:
                lora_layer = LoRALinearLayer(m.in_features, m.out_features, rank=lora_rank)
                loras.append(lora_layer)
            self.loras_dict.update({task: loras})

        # replace linear with lora linear
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear) and 'loras_dict' not in n:
                # define lora linear
                lora_linear = LoRACompatibleLinear(m.in_features, m.out_features)
                # copy weight and bias
                lora_linear.weight.data.copy_(m.weight.data)
                if hasattr(m, 'bias') and m.bias is not None:
                    lora_linear.bias.data.copy_(m.bias.data)  # type: ignore
                else:
                    lora_linear.bias = None
                # replace linear with lora linear
                parent = self
                *path, name = n.split('.')
                while path:
                    parent = parent.get_submodule(path.pop(0))
                parent._modules[name] = lora_linear

    def forward(self, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        outs = []

        h = hint.type(self.dtype)
        for module, zero_conv in zip(self.input_blocks, self.zero_convs):
            h = module(h, emb, context)
            outs.append(zero_conv(h, emb, context))

        h = self.middle_block(h, emb, context)
        outs.append(self.middle_block_out(h, emb, context))

        return outs

    def switch_lora(self, task: str):
        assert task in self.tasks
        lora = self.loras_dict[task]
        idx = 0
        for n, m in self.named_modules():
            if isinstance(m, LoRACompatibleLinear):
                m.set_lora_layer(lora[idx])
                idx += 1


class ControlPretrainLDM(ControlLDM):

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c_dict = super().get_input(batch, k, bs, *args, **kwargs)
        task = batch['task'][0][8:]
        c_dict.update({'task': task})
        return x, c_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            self.control_model.switch_lora(cond['task'])
            hint = torch.cat(cond['c_concat'], 1)
            hint = self.get_first_stage_encoding(self.encode_first_stage(hint))
            control = self.control_model(hint=hint, timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    @torch.no_grad()
    def log_images(self, batch, N=4, n_row=2, sample=False, ddim_steps=50, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=False, unconditional_guidance_scale=9.0, unconditional_guidance_label=None,
                   use_ema_scope=True,
                   **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=N)
        c_cat, c, task = c["c_concat"][0][:N], c["c_crossattn"][0][:N], c["task"]
        N = min(z.shape[0], N)
        n_row = min(z.shape[0], n_row)
        log["reconstruction"] = self.decode_first_stage(z)
        log["control"] = c_cat * 2.0 - 1.0
        log["conditioning"] = log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = einops.repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = einops.rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = einops.rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            samples, z_denoise_row = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "task": task},
                                                     batch_size=N, ddim=use_ddim,
                                                     ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        if unconditional_guidance_scale > 1.0:
            uc_cross = self.get_unconditional_conditioning(N)
            uc_cat = c_cat  # torch.zeros_like(c_cat)
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross], "task": task}
            samples_cfg, _ = self.sample_log(cond={"c_concat": [c_cat], "c_crossattn": [c], "task": task},
                                             batch_size=N, ddim=use_ddim,
                                             ddim_steps=ddim_steps, eta=ddim_eta,
                                             unconditional_guidance_scale=unconditional_guidance_scale,
                                             unconditional_conditioning=uc_full,
                                             )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        opt = torch.optim.AdamW(params, lr=lr)
        print(f'Optimizable params: {sum([p.numel() for p in params])/1e6:.1f}M')
        return opt
