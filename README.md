<div align="center">
<h1><span style="color:#93cf6a;">M</span>ulti-<span style="color:#93cf6a;">v</span>iew <span style="color:#93cf6a;">P</span>yramid Transformer: Look Coarser to See Broader</h1>

<a href="https://arxiv.org/abs/2512.07806"><img src="https://img.shields.io/badge/arXiv-2512.07806-b31b1b" alt="arXiv"></a>
<a href="https://gynjn.github.io/MVP/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

[Gyeongjin Kang](https://gynjn.github.io/info/), [Seungkwon Yang](https://github.com/yang-gwon2), [Seungtae Nam](https://github.com/stnamjef), [Younggeun Lee](https://github.com/Younggeun-L), [Jungwoo Kim](https://github.com/jungcow), [Eunbyung Park](https://silverbottlep.github.io/index.html)
</div>

Official repo for the paper "**Multi-view Pyramid Transformer: Look Coarser to See Broader**"

## Installation

```bash
# create conda environment
conda create -n mvp python=3.11 -y
conda activate mvp

# install PyTorch (adjust cuda version according to your system)
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

## Checkpoints
The model checkpoints are host on [HuggingFace](https://huggingface.co/Gynjn/MVP) ([mvp_960x540](https://huggingface.co/Gynjn/MVP/resolve/main/mvp.pt?download=true)).


For training and evaluation, we used the DL3DV dataset after applying undistortion preprocessing with this [script](https://github.com/arthurhero/Long-LRM/blob/main/data/prosess_dl3dv.py), originally introduced in [Long-LRM](https://arthurhero.github.io/projects/llrm/index.html). 

Download the DL3DV benchmark dataset from [here](https://huggingface.co/datasets/DL3DV/DL3DV-Benchmark/tree/main), and apply undistortion preprocessing.

## Inference

Update the `inference.ckpt_path` field in `configs/inference.yaml` with the pretrained model.

Update the entries in `data/dl3dv_eval.txt` to point to the correct processed dataset path.

```bash
# inference
CUDA_VISIBLE_DEVICES=0 python inference.py --config configs/inference.yaml
```

## Train

Update the `configs/api_keys.yaml` with your own personal wandb api key.

Update the entries in `data/dl3dv_train.txt` to point to the correct processed dataset path.

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
