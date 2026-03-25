<div align="center">
<h1><span style="color:#93cf6a;">M</span>ulti-<span style="color:#93cf6a;">v</span>iew <span style="color:#93cf6a;">P</span>yramid Transformer: Look Coarser to See Broader</h1>

<a href="https://arxiv.org/abs/2512.07806"><img src="https://img.shields.io/badge/arXiv-2512.07806-b31b1b" alt="arXiv"></a>
<a href="https://gynjn.github.io/MVP/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

[Gyeongjin Kang](https://gynjn.github.io/info/), [Seungkwon Yang](https://github.com/yang-gwon2), [Seungtae Nam](https://github.com/stnamjef), [Younggeun Lee](https://github.com/Younggeun-L), [Jungwoo Kim](https://github.com/jungcow), [Eunbyung Park](https://silverbottlep.github.io/index.html)
</div>

Official repo for the paper "**Multi-view Pyramid Transformer: Look Coarser to See Broader**"

## News

**[Mar 2026]** MVP optionally supports [Flash Attention 4](https://github.com/Dao-AILab/flash-attention) for faster attention computation. If `flash-attn-4` is installed, it will be used automatically; otherwise the code falls back to the standard `F.scaled_dot_product_attention`.

| Views | H100 + FA3 (s) | B200 + FA4 (s) | Speedup |
|------:|---------------:|---------------:|--------:|
|    16 |           0.09 |           0.05 |   1.8×  |
|    32 |           0.17 |           0.10 |   1.7×  |
|    64 |           0.36 |           0.20 |   1.8×  |
|   128 |           0.77 |           0.43 |   1.8×  |
|   192 |           1.23 |           0.70 |   1.8×  |
|   256 |           1.84 |           1.08 |   1.7×  |

*Reconstruction time (seconds). H100 numbers from the original paper.*

**[Mar 2026]** We've updated the codebase with a CUDA implementation of Opacity with SH coefficients, reducing both training time and memory consumption. Kudos to [Hyeongbhin-Cho](https://github.com/Hyeongbhin-Cho) for the contribution. See details in the original [repo](https://github.com/Hyeongbhin-Cho/gsplat_vlab).

**[Mar 2026]** We've released [2Xplat](https://hwasikjeong.github.io/2Xplat/), a pose-free version of MVP built on a two-expert design — one for camera pose estimation, one for 3DGS generation. It outperforms prior pose-free methods and matches state-of-the-art posed approaches in under 5K training iterations.

## Installation

```bash
# 1. Clone the repository
# If starting fresh (clone everything at once):
git clone --recurse-submodules https://github.com/Gynjn/MVP.git

# If already cloned (initialize submodules):
git submodule update --init --recursive

# 2. Create and activate conda environment
conda create -n mvp python=3.11 -y
conda activate mvp

# 3. Install dependencies (adjust CUDA version in requirements.txt to match your system)
pip install -r requirements.txt

# 4. Install CUDA kernels
cd rendering_cuda
pip install . --no-build-isolation
cd ../sh_cuda
pip install . --no-build-isolation

# 5. Optional
pip install flash-attn-4
```

## Checkpoints
The model checkpoints are host on [HuggingFace](https://huggingface.co/Gynjn/MVP) ([mvp_960x540](https://huggingface.co/Gynjn/MVP/resolve/main/mvp.pt?download=true)).


For training and evaluation, we used the DL3DV dataset after applying undistortion preprocessing with this [script](https://github.com/arthurhero/Long-LRM/blob/main/data/prosess_dl3dv.py), originally introduced in [Long-LRM](https://arthurhero.github.io/projects/llrm/index.html). 

Download the DL3DV benchmark dataset from [here](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/tree/main), and apply undistortion preprocessing.

For benchmark data, we provide preprocessed version originally sourced from the [RayZer](https://github.com/hwjiang1510/RayZer) repository. You can find the preprocessed data [here](https://huggingface.co/datasets/hwjiang/DL3DV-benchmark-preprocessed/resolve/main/dl3dv_benchmark.zip?download=true). Thanks to [Hanwen Jiang](https://hwjiang1510.github.io/) for sharing the preprocessed data.

## Inference

Update the `inference.ckpt_path` field in `configs/inference.yaml` with the pretrained model.

Update the entries in `data/dl3dv_benchmark.txt` to point to the correct processed dataset path.

```bash
# inference
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/inference.yaml
```

## Train

Update the `configs/api_keys.yaml` with your own personal wandb api key.

Update the entries in `data/dl3dv_train.txt` to point to the correct processed dataset path.

If you have enough GPU memory, disable gradient checkpointing in each stage function `run_stage1`, `run_stage2`, and `run_stage3` in `model/mvp.py`.

```bash
# Example for single GPU training
CUDA_VISIBLE_DEVICES=0 python train_single.py --config configs/train_stage1.yaml

# Example for multi GPU training
torchrun --nproc_per_node 8 --nnodes 1 \
         --rdzv_id 1234 --rdzv_endpoint localhost:8888 \
         train.py --config configs/train_stage1.yaml
```


## TODO List
- [ ] Preprocessed Tanks&Temple and Mip-NeRF360 dataset

## Citation
```
@article{kang2025multi,
  title={Multi-view Pyramid Transformer: Look Coarser to See Broader},
  author={Kang, Gyeongjin and Yang, Seungkwon and Nam, Seungtae and Lee, Younggeun and Kim, Jungwoo and Park, Eunbyung},
  journal={arXiv preprint arXiv:2512.07806},
  year={2025}
}
```

## Related project

This project is built on many amazing research works, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [gsplat](https://github.com/nerfstudio-project/gsplat)
- [LVSM](https://github.com/haian-jin/LVSM)
- [Long-LRM](https://github.com/arthurhero/Long-LRM)
- [LaCT](https://github.com/a1600012888/LaCT)
- [iLRM](https://github.com/Gynjn/iLRM)
- [ProPE](https://github.com/liruilong940607/prope)
- [LVT](https://toobaimt.github.io/lvt/)

## ACKNOWLEDGEMENTS

This work was supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2025-02653113, High-Performance Research AI Computing Infrastructure Support at the 2 PFLOPS Scale)
