import torch
from torch import nn

def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def swin_adapt_position_encoding(m, after_image_size=384):
    new_window_size = after_image_size / 32
    # assume model size = 224

    # relative_position_index_keys = [n for n,p in m.named_buffers() if "relative_position_index" in n]
    for l in m.layers:
        for x in l.blocks:
            coords_h = torch.arange(new_window_size)
            coords_w = torch.arange(new_window_size)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += new_window_size - 1  # shift to start from 0
            relative_coords[:, :, 1] += new_window_size - 1
            relative_coords[:, :, 0] *= 2 * new_window_size - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            x.attn.register_buffer("relative_position_index", relative_position_index)

    # attn_mask_keys = [n for n,p in m.named_buffers() if "attn_mask" in n]
    for l in m.layers:
        for x in l.blocks:
            if hasattr(x, 'attn_mask') and x.attn_mask is not None:
                H, W = after_image_size, after_image_size
                img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
                cnt = 0
                for h in (
                        slice(0, -new_window_size),
                        slice(-new_window_size, -new_window_size),
                        slice(-new_window_size, None)):
                    for w in (
                            slice(0, -new_window_size),
                            slice(-new_window_size, -new_window_size),
                            slice(-new_window_size, None)):
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                mask_windows = window_partition(img_mask, new_window_size)  # num_win, window_size, window_size, 1
                mask_windows = mask_windows.view(-1, new_window_size * new_window_size)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
                x.register_buffer("attn_mask", attn_mask)
    

    # relative_position_bias_table_keys = [n for n,p in m.named_parameters() if "relative_position_bias_table" in n]
    
    # # absolute_pos_embed_keys = [n for n,p in m.named_parameters() if "absolute_pos_embed" in n]