"""
Convert a 5-channel ActorCriticSpatial checkpoint to the 6-channel
architecture with the motion-trail channel.

The only shape change is `dynamic_cnn.0.weight`:
    old: (16, 3, 5, 5)   →   new: (16, 4, 5, 5)

The 4th input slot (motion channel) is initialised to zero so the
forward pass at fine-tune step 0 is bit-identical to the 5-channel
checkpoint. All other tensors copy unchanged.

Usage:
    python -m test_rl_fast.models.convert_5ch_to_6ch \
        /home/efe/ros2_ws/rl_osl/efe_0_2_wall_99975168.pt \
        /home/efe/ros2_ws/efe_0_2_wall_99975168_6ch.pt
"""

import argparse
import sys

import torch


KEY = 'dynamic_cnn.0.weight'


def convert(src_path: str, dst_path: str, verify: bool = True) -> None:
    print(f'Loading 5-channel checkpoint from {src_path}')
    ckpt = torch.load(src_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' not in ckpt:
        sys.exit('Checkpoint has no model_state_dict — wrong format?')

    sd = ckpt['model_state_dict']
    if KEY not in sd:
        sys.exit(f'Missing key {KEY} — checkpoint not a spatial model?')

    old_w = sd[KEY]
    if old_w.ndim != 4:
        sys.exit(f'{KEY} has unexpected ndim={old_w.ndim}')
    if old_w.shape[1] == 4:
        print(f'{KEY} already has 4 input channels; nothing to do.')
        return
    if old_w.shape[1] != 3:
        sys.exit(f'{KEY} has {old_w.shape[1]} input channels; expected 3.')

    out_c, _, kh, kw = old_w.shape
    new_w = torch.zeros((out_c, 4, kh, kw), dtype=old_w.dtype)
    new_w[:, :3] = old_w   # copy existing weights into channels 0..2
    # 4th channel weights remain zero — motion channel contributes 0 to layer
    # output until gradient flows in during fine-tuning.

    sd[KEY] = new_w
    print(f'  {KEY}: {tuple(old_w.shape)} → {tuple(new_w.shape)}  (4th slot = 0)')

    # Sanity: max |Δ| of every other tensor is 0 (we didn't touch them).
    if verify:
        sd_orig = torch.load(src_path, map_location='cpu',
                             weights_only=False)['model_state_dict']
        for k, v in sd.items():
            if k == KEY:
                continue
            if not torch.equal(v, sd_orig[k]):
                sys.exit(f'Tensor {k} unexpectedly changed.')
        print('  Other tensors verified unchanged.')

    print(f'Saving 6-channel checkpoint to {dst_path}')
    torch.save(ckpt, dst_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('src', help='5-channel checkpoint path')
    p.add_argument('dst', help='Output path for 6-channel checkpoint')
    p.add_argument('--no-verify', action='store_true')
    args = p.parse_args()
    convert(args.src, args.dst, verify=not args.no_verify)


if __name__ == '__main__':
    main()
