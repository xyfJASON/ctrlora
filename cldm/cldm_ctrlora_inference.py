import copy

import torch
import torch.nn as nn

from cldm.ddim_hacked import DDIMSampler
from cldm.cldm import ControlNet, ControlLDM
from cldm.lora import LoRALinearLayer, LoRACompatibleLinear
from cldm.switchable import SwitchableConv2d, SwitchableLayerNorm, SwitchableGroupNorm
from ldm.modules.diffusionmodules.util import timestep_embedding
import ipdb

class ControlNetInference(ControlNet):
    def __init__(self, lora_rank=128, lora_num=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_rank = lora_rank
        self.lora_num = lora_num

        # delete input hint block
        del self.input_hint_block

        # define a list of lora layers
        self.loras_list = nn.ModuleList([])
        linear_modules = [m for n, m in self.named_modules() if isinstance(m, nn.Linear)]
        for _ in range(self.lora_num):
            loras = nn.ModuleList([])
            for m in linear_modules:
                lora_layer = LoRALinearLayer(m.in_features, m.out_features, rank=lora_rank)
                loras.append(lora_layer)
            self.loras_list.append(loras)

        # define a list of zero convs layers
        self.zero_convs_list = nn.ModuleList([])
        zero_convs_modules = [m for n, m in self.named_modules() if ('zero_convs' in n or 'middle_block_out' in n) and isinstance(m, nn.Conv2d)]
        for _ in range(self.lora_num):
            zero_convs = nn.ModuleList([])
            for m in zero_convs_modules:
                zero_convs.append(copy.deepcopy(m))
            self.zero_convs_list.append(zero_convs)

        # define a list of norm layers
        self.norms_list = nn.ModuleList([])
        norm_modules = [m for n, m in self.named_modules() if 'norm' in n and isinstance(m, (nn.GroupNorm, nn.LayerNorm))]
        for _ in range(self.lora_num):
            norms = nn.ModuleList([])
            for m in norm_modules:
                norms.append(copy.deepcopy(m))
            self.norms_list.append(norms)

        # replace
        for n, m in self.named_modules():
            if 'loras_list' in n or 'zero_convs_list' in n or 'norms_list' in n:
                continue

            # replace linear with lora linear
            if isinstance(m, nn.Linear):
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

            # replace zero convs with switchable conv
            elif ('zero_convs' in n or 'middle_block_out' in n) and isinstance(m, nn.Conv2d):
                # define switchable conv
                switchable_conv = SwitchableConv2d(
                    m.in_channels, m.out_channels, m.kernel_size, m.stride,
                    m.padding, m.dilation, m.groups, m.bias is not None,
                )
                # replace conv with switchable conv
                parent = self
                *path, name = n.split('.')
                while path:
                    parent = parent.get_submodule(path.pop(0))
                parent._modules[name] = switchable_conv

            # replace norm with switchable norm
            elif 'norm' in n and isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                # define switchable norm
                if isinstance(m, nn.GroupNorm):
                    switchable_norm = SwitchableGroupNorm(m.num_groups, m.num_channels)
                else:
                    switchable_norm = SwitchableLayerNorm(m.normalized_shape, m.eps, m.elementwise_affine)
                # replace norm with switchable norm
                parent = self
                *path, name = n.split('.')
                while path:
                    parent = parent.get_submodule(path.pop(0))
                parent._modules[name] = switchable_norm

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

    def switch_lora(self, index: int):
        lora = self.loras_list[index]
        zero_convs = self.zero_convs_list[index]
        norms = self.norms_list[index]
        idx, idx_zero, idx_norm = 0, 0, 0
        for n, m in self.named_modules():
            if isinstance(m, LoRACompatibleLinear):
                m.set_lora_layer(lora[idx])  # type: ignore
                idx += 1
            elif isinstance(m, SwitchableConv2d):
                m.set_conv_layer(zero_convs[idx_zero])  # type: ignore
                idx_zero += 1
            elif isinstance(m, (SwitchableGroupNorm, SwitchableLayerNorm)):
                m.set_norm_layer(norms[idx_norm])  # type: ignore
                idx_norm += 1

    def copy_weights_to_switchable(self):
        """
        Clumsy workaround to store the weights to the switchable layers,
        Need to be called after switch_lora() and load_state_dict().
        """
        for n, m in self.named_modules():
            if isinstance(m, (SwitchableConv2d, SwitchableGroupNorm, SwitchableLayerNorm)):
                m.copy_weights()


class ControlInferenceLDM(ControlLDM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_weights = [1.0 / self.control_model.lora_num] * self.control_model.lora_num

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        ddim_sampler = DDIMSampler(self)
        b, c, h, w = cond["c_concat"][0].shape
        shape = (self.channels, h // 8, w // 8)
        samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)
        return samples, intermediates

    def apply_model(self, x_noisy, t, conds, *args, **kwargs):
        if isinstance(conds, dict):
            conds = [conds]
        assert isinstance(conds, (list, tuple))
        assert len(conds) == self.control_model.lora_num
        assert len(self.lora_weights) == self.control_model.lora_num
        weights = self.lora_weights

        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(conds[0]['c_crossattn'], 1)
        if conds[0]['c_ip'][0] is not None:
            cond_ip = torch.cat(conds[0]['c_ip'], 1)
        else:
            cond_ip=None
        if conds[0]['c_concat'][0] is not None:
            controls = []
            for i, cond in enumerate(conds): 
                self.control_model.switch_lora(i)
                hint = torch.cat(cond['c_concat'], 1)
                hint = self.get_first_stage_encoding(self.encode_first_stage(hint))
                control = self.control_model(hint=hint, timesteps=t, context=cond_txt)
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                controls.append(control)
            control = [c * weights[0] for c in controls[0]]
            for i in range(1, len(controls)):
                control = [c + controls[i][j] * weights[i] for j, c in enumerate(control)]
        else:
            control = None
        if isinstance(cond_txt, list):
            context_with_ip = [[txt, cond_ip] for txt in cond_txt]
        else:
            context_with_ip = [[cond_txt, cond_ip]]
        eps = diffusion_model(x=x_noisy, timesteps=t, context=context_with_ip, control=control, only_mid_control=self.only_mid_control)
        return eps
