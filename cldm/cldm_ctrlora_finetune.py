import torch
import torch.nn as nn

from cldm.ddim_hacked import DDIMSampler
from cldm.cldm import ControlNet, ControlLDM
from cldm.lora import LoRALinearLayer, LoRACompatibleLinear
from ldm.modules.diffusionmodules.util import timestep_embedding


class ControlNetFinetune(ControlNet):
    def __init__(self, ft_with_lora=True, lora_rank=128, norm_trainable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ft_with_lora = ft_with_lora
        self.lora_rank = lora_rank
        self.norm_trainable = norm_trainable

        # delete input hint block
        del self.input_hint_block

        if ft_with_lora:
            # replace linear with lora linear
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    # define lora linear
                    lora_layer = LoRALinearLayer(m.in_features, m.out_features, rank=lora_rank)
                    lora_linear = LoRACompatibleLinear(m.in_features, m.out_features, lora_layer=lora_layer)
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


class ControlFinetuneLDM(ControlLDM):

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)

        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            hint = torch.cat(cond['c_concat'], 1)
            hint = self.get_first_stage_encoding(self.encode_first_stage(hint))
            control = self.control_model(hint=hint, timesteps=t, context=cond_txt)
            control = [c * scale for c, scale in zip(control, self.control_scales)]
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)

        return eps

    def configure_optimizers(self):
        lr = self.learning_rate
        params = []
        f = open('./tmp/finetune_trainable_params.txt', 'w')
        for n, p in self.control_model.named_parameters():
            assert 'input_hint' not in n
            if self.control_model.ft_with_lora:  # only train lora layers, zero convs and norm layers
                if 'lora_layer' in n:  # lora layers
                    params.append(p)
                    f.write(n + '\n')
                elif 'zero_convs' in n or 'middle_block_out' in n:  # zero convs
                    # note that middle_block_out is also a zero conv added by controlnet!
                    params.append(p)
                    f.write(n + '\n')
                elif 'norm' in n and self.control_model.norm_trainable:  # norm layers
                    params.append(p)
                    f.write(n + '\n')
            else:
                assert 'lora_layer' not in n
                params.append(p)
                f.write(n + '\n')
        opt = torch.optim.AdamW(params, lr=lr)
        print(f'Optimizable params: {sum([p.numel() for p in params])/1e6:.1f}M')
        f.close()
        return opt
