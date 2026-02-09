# Copyright (c) 2024, Ziwen Chen.

import os
import json
import random
import traceback
import numpy as np
import PIL.Image as Image
import cv2
import torch
import torch.nn.functional as F
from easydict import EasyDict as edict
# from torch.utils.data import Dataset
from data.base_dataset import BaseDataset
import torchvision.transforms as transforms
from einops import repeat

def resize_and_crop(image, target_size, fxfycxcy):
    """
    Resize and crop image to target_size, adjusting camera parameters accordingly.
    
    Args:
        image: PIL Image
        target_size: (width, height) tuple
        fxfycxcy: [fx, fy, cx, cy] list
    
    Returns:
        tuple: (resized_cropped_image, adjusted_fxfycxcy)
    """
    target_width, target_height = target_size
    
    fx, fy, cx, cy, h, w = fxfycxcy
    
    resized_image = cv2.resize(image, (target_width, target_height))
    new_fx = fx * (target_width / w)
    new_fy = fy * (target_height / h)
    new_cx = cx * (target_width / w)
    new_cy = cy * (target_height / h)
        
    return resized_image, [new_fx, new_fy, new_cx, new_cy]

class Dataset(BaseDataset):
    def __init__(self, config):
        super().__init__(config)        
        self.config = config
        self.evaluation = config.get("evaluation", False)
        if self.evaluation and "data_eval" in config:
            self.config.data.update(config.data_eval)
        data_path_text = config.data.data_path
        data_folder = data_path_text.rsplit('/', 1)[0] 
        with open(data_path_text, 'r') as f:
            self.data_path = f.readlines()
        self.data_path = [x.strip() for x in self.data_path]
        self.data_path = [x for x in self.data_path if len(x) > 0]
        for i, data_path in enumerate(self.data_path):
            if not data_path.startswith("/"):
                self.data_path[i] = os.path.join(data_folder, data_path)
        self._scene_len()

    def _scene_len(self):
        self.num_of_scenes = len(self.data_path)

    def process_frames(self, frames, image_base_dir, resolution):
        fxfycxcy_list = []
        image_list = []

        resize_w, resize_h = int(resolution[1]), int(resolution[0])

        for frame in frames:
            image = np.array(Image.open(os.path.join(image_base_dir, frame["file_path"])))
            fxfycxcyhw = [frame["fx"], frame["fy"], frame["cx"], frame["cy"], frame["h"], frame["w"]]

            image, fxfycxcy = resize_and_crop(image, (resize_w, resize_h), fxfycxcyhw)

            fxfycxcy_list.append(fxfycxcy)
            image_list.append(torch.from_numpy(image / 255.0).permute(2, 0, 1).float())  # (3, resize_h, resize_w)

        intrinsics = torch.tensor(fxfycxcy_list, dtype=torch.float32)  # (num_frames, 4)
        images = torch.stack(image_list, dim=0)
        c2ws = np.stack([np.array(frame["w2c"]) for frame in frames])
        c2ws = np.linalg.inv(c2ws)  # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        
        c2w_bucket = repeat(torch.eye(4, dtype=torch.float32), 'h w -> b h w', b=c2ws.shape[0]).clone()
        c2w_bucket[:, :3] = c2ws[:, :3]  # (num_frames, 4, 4)

        return images, intrinsics, c2w_bucket

    def _get_views(self, sampled_idx, resolution, num_views_to_input, num_views_to_target):
        data_path = self.data_path[sampled_idx]
        data_json = json.load(open(data_path, 'r'))
        scene_name = data_json['scene_name']
        frames = data_json['frames']
        image_base_dir = data_path.rsplit('/', 1)[0]

        # read config
        input_frame_select_type = self.config.data.input_frame_select_type
        target_frame_select_type = self.config.data.target_frame_select_type

        target_has_input = self.config.data.target_has_input
        min_frame_dist = self.config.data.min_frame_dist
        max_frame_dist = self.config.data.max_frame_dist
        if min_frame_dist == "all":
            min_frame_dist = len(frames) - 1
            max_frame_dist = min_frame_dist
        min_frame_dist = min(min_frame_dist, len(frames) - 1)
        max_frame_dist = min(max_frame_dist, len(frames) - 1)
        assert min_frame_dist <= max_frame_dist
        if target_has_input:
            assert min_frame_dist >= max(num_views_to_input, num_views_to_target) - 1
        else:
            assert min_frame_dist >= num_views_to_input + num_views_to_target - 1
        frame_dist = np.random.randint(min_frame_dist, max_frame_dist + 1)
        shuffle_input_prob = self.config.data.get("shuffle_input_prob", 0.0)
        shuffle_input = np.random.rand() < shuffle_input_prob
        reverse_input_prob = self.config.data.get("reverse_input_prob", 0.0)
        reverse_input = np.random.rand() < reverse_input_prob

        # get frame range
        start_frame_idx = np.random.randint(0, len(frames) - frame_dist)
        end_frame_idx = start_frame_idx + frame_dist
        frame_idx = list(range(start_frame_idx, end_frame_idx + 1))
    
        # get target frames
        if target_frame_select_type == 'random':
            target_frame_idx = np.random.choice(frame_idx, num_views_to_target, replace=False)
        elif target_frame_select_type == 'uniform':
            target_frame_idx = np.linspace(start_frame_idx, end_frame_idx, num_views_to_target, dtype=int)
        elif target_frame_select_type == 'uniform_every':
            uniform_every = self.config.data.target_uniform_every
            target_frame_idx = list(range(start_frame_idx, end_frame_idx + 1, uniform_every))
            num_views_to_target = len(target_frame_idx)
        else:
            raise NotImplementedError
        target_frame_idx = sorted(target_frame_idx)

        # get input frames
        if not target_has_input:
            frame_idx = [x for x in frame_idx if x not in target_frame_idx]
        if input_frame_select_type == 'random':
            input_frame_idx = np.random.choice(frame_idx, num_views_to_input, replace=False)
        elif input_frame_select_type == 'uniform':
            input_frame_idx = np.linspace(0, len(frame_idx) - 1, num_views_to_input, dtype=int)
            input_frame_idx = [frame_idx[i] for i in input_frame_idx]
        else:
            raise NotImplementedError
        input_frame_idx = sorted(input_frame_idx)
        if reverse_input:
            input_frame_idx = input_frame_idx[::-1]
        if shuffle_input:
            np.random.shuffle(input_frame_idx)

        target_frames = [frames[i] for i in target_frame_idx]
        target_images, target_intr, target_c2ws = self.process_frames(target_frames, image_base_dir, resolution)
    
        input_frames = [frames[i] for i in input_frame_idx]
        input_images, input_intr, input_c2ws = self.process_frames(input_frames, image_base_dir, resolution)

        if (target_c2ws[:, :3, 3] > 1e3).any():
            print(f"encounter large translation in target poses: {target_c2ws[:, :3, 3].max()}")
            assert False
        if (input_c2ws[:, :3, 3] > 1e3).any():
            print(f"encounter large translation in input poses: {input_c2ws[:, :3, 3].max()}")
            assert False

        if any(torch.isnan(torch.det(target_c2ws[:, :3, :3]))):
            print(f"encounter nan in target poses: {target_c2ws[:, :3, :3]}")
            assert False
        if any(torch.isnan(torch.det(input_c2ws[:, :3, :3]))):
            print(f"encounter nan in input poses: {input_c2ws[:, :3, :3]}")
            assert False

        if not torch.allclose(torch.det(target_c2ws[:, :3, :3]), torch.det(target_c2ws[:, :3, :3]).new_tensor(1.0)):
            print(f"det of target poses not equal to 1")
            assert False
        if not torch.allclose(torch.det(input_c2ws[:, :3, :3]), torch.det(input_c2ws[:, :3, :3]).new_tensor(1.0)):
            print(f"det of input poses not equal to 1")
            assert False

        # normalize input camera poses
        position_avg = input_c2ws[:, :3, 3].mean(0) # (3,)
        forward_avg = input_c2ws[:, :3, 2].mean(0) # (3,)
        down_avg = input_c2ws[:, :3, 1].mean(0) # (3,)
        # gram-schmidt process
        forward_avg = F.normalize(forward_avg, dim=0)
        down_avg = F.normalize(down_avg - down_avg.dot(forward_avg) * forward_avg, dim=0)
        right_avg = torch.cross(down_avg, forward_avg)
        pos_avg = torch.stack([right_avg, down_avg, forward_avg, position_avg], dim=1) # (3, 4)
        pos_avg = torch.cat([pos_avg, torch.tensor([[0, 0, 0, 1]], device=pos_avg.device).float()], dim=0) # (4, 4)
        pos_avg_inv = torch.inverse(pos_avg)

        input_c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), input_c2ws)
        target_c2ws = torch.matmul(pos_avg_inv.unsqueeze(0), target_c2ws)
    
        # scale scene size
        position_max = input_c2ws[:, :3, 3].abs().max()
        scene_scale = self.config.data.get("scene_scale", 1.0) * position_max
        scene_scale = 1.0 / scene_scale

        input_c2ws[:, :3, 3] *= scene_scale
        target_c2ws[:, :3, 3] *= scene_scale

        if torch.isnan(input_c2ws).any() or torch.isinf(input_c2ws).any():
            print("encounter nan or inf in input poses")
            assert False

        if torch.isnan(target_c2ws).any() or torch.isinf(target_c2ws).any():
            print("encounter nan or inf in target poses")
            assert False
    
        image = torch.cat([input_images, target_images], dim=0)
        fxfycxcy = torch.cat([input_intr, target_intr], dim=0)
        c2w = torch.cat([input_c2ws, target_c2ws], dim=0)
        input_indices = torch.tensor(input_frame_idx).long().unsqueeze(-1)
        target_indices = torch.tensor(target_frame_idx).long().unsqueeze(-1)
        # image_indices = input_frame_idx + target_frame_idx
        # image_indices = torch.tensor(image_indices).long().unsqueeze(-1)

        ret_dict = {
            "input_image": input_images,  # (num_input, 3, resize_h, resize_w)
            "input_fxfycxcy": input_intr,  # (num_input, 4)
            "input_c2w": input_c2ws,  # (num_input,
            "target_image": target_images,  # (num_target, 3, resize_h, resize_w)
            "target_fxfycxcy": target_intr,  # (num_target, 4)
            "target_c2w": target_c2ws,  # (num_target,
            "input_indices": input_indices,
            "target_indices": target_indices,
            # "image": image,  # (num_input + num_target, 3, resize_h, resize_w)
            # "fxfycxcy": fxfycxcy,  # (num_input + num_target, 4)
            # "c2w": c2w,  # (num_input + num_target, 4, 4)
            # "index": image_indices,
            "scene_name": scene_name,
        }

        return ret_dict


if __name__ == "__main__":
    # test dataset
    config = edict()
    config.data = edict()
    config.model = edict()
    config.data.data_path = "example_data/mydesk.txt"
    config.data.resize_h = 128
    config.data.resize_w = 416
    config.model.patch_size = 16
    config.data.square_crop = True
    config.data.input_frame_select_type = "kmeans"
    config.data.target_frame_select_type = "uniform_every"
    config.data.num_input_frames = 32
    config.data.num_target_frames = 8
    config.data.target_has_input = False
    config.data.min_frame_dist = "all"
    config.data.max_frame_dist = 64
    config.data.target_uniform_every = 8

    dataset = Dataset(config)
    print("dataset length:", len(dataset))

    for i in range(len(dataset)):
        data = dataset[i]
        print("scene_name:", data["scene_name"])
        print("input_images:", data["input_images"].shape)
        print("input_intr:", data["input_intr"].shape)
        print("input_c2ws:", data["input_c2ws"].shape)
        print("target_images:", data["test_images"].shape)
        print("target_intr:", data["test_intr"].shape)
        print("target_c2ws:", data["test_c2ws"].shape)
        print("pos_avg_inv:", data["input_pos_avg_inv"].shape)
        print("scene_scale:", data["scene_scale"])


