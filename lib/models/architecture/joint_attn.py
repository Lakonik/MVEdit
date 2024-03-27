import torch
import torch.nn as nn


class CrossImageAttnProcWrapper(nn.Module):

    def __init__(self, base_attn_proc):
        super().__init__()
        self.base_attn_proc = base_attn_proc

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 num_cross_attn_imgs=1):
        if num_cross_attn_imgs > 1:
            hidden_states = hidden_states.view(
                hidden_states.shape[0] // num_cross_attn_imgs,
                num_cross_attn_imgs * hidden_states.shape[1],
                hidden_states.shape[2])
            if encoder_hidden_states is not None:  # cross attn to text
                encoder_hidden_states_ = encoder_hidden_states.view(
                    encoder_hidden_states.shape[0] // num_cross_attn_imgs,
                    num_cross_attn_imgs,
                    encoder_hidden_states.shape[1],
                    encoder_hidden_states.shape[2])
                encoder_hidden_states = encoder_hidden_states_.mean(dim=1)
                # assert (encoder_hidden_states == encoder_hidden_states_[:, 0]).all()
            if attention_mask is not None:
                raise NotImplementedError
            if temb is not None:
                raise NotImplementedError
        hidden_states = self.base_attn_proc(
            attn, hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask, temb=temb)
        if num_cross_attn_imgs > 1:
            hidden_states = hidden_states.view(
                hidden_states.shape[0] * num_cross_attn_imgs,
                hidden_states.shape[1] // num_cross_attn_imgs,
                hidden_states.shape[2])
        return hidden_states


def apply_cross_image_attn_proc(model):
    attn_procs = dict()
    for name in model.attn_processors.keys():
        attn_procs[name] = CrossImageAttnProcWrapper(model.attn_processors[name])
    model.set_attn_processor(attn_procs)
    return
