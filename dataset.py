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
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from einops import repeat

def resize_and_crop(image, target_size, fxfycxcy, square_crop):

    target_width, target_height = target_size
    
    fx, fy, cx, cy, h, w = fxfycxcy
    
    resized_image = cv2.resize(image, (target_width, target_height))
    new_fx = fx * (target_width / w)
    new_fy = fy * (target_height / h)
    new_cx = cx * (target_width / w)
    new_cy = cy * (target_height / h)
        
    # squre crop
    if square_crop:
        min_size = min(target_width, target_height)
        start_h = (target_height - min_size) // 2
        start_w = (target_width - min_size) // 2
        resized_image = resized_image[start_h:start_h+min_size, start_w:start_w+min_size, :]
        new_cx -= start_w
        new_cy -= start_h

    return resized_image, [new_fx, new_fy, new_cx, new_cy]

class Dataset(Dataset):
    def __init__(self, config):
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
        self.json_path = config.data.get("json_path", None)
        self.random_crop = self.config.data.get("random_crop", 1.0)

    def __len__(self):
        return len(self.data_path)

    def process_frames(self, frames, image_base_dir, random_crop_ratio=None):
        fxfycxcy_list = []
        image_list = []
        c2w_list = []

        resize_h = self.config.data.get("resize_h", -1)
        patch_size = self.config.model.patch_size * self.config.get("patch_factor", 2)
        square_crop = self.config.data.square_crop


        resize_w = int(resize_h / 540 * 960) # 455.1
        resize_h = int(round(resize_h / patch_size)) * patch_size # 256
        resize_w = int(round(resize_w / patch_size)) * patch_size # 448

        for frame in frames:
            image = np.array(Image.open(os.path.join(image_base_dir, frame["file_path"])))
            fxfycxcyhw = [frame["fx"], frame["fy"], frame["cx"], frame["cy"], frame["h"], frame["w"]]

            image, fxfycxcy = resize_and_crop(image, (resize_w, resize_h), fxfycxcyhw, square_crop)

            fxfycxcy_list.append(fxfycxcy)
            image_list.append(torch.from_numpy(image / 255.0).permute(2, 0, 1).float())  # (3, resize_h, resize_w)

        intrinsics = torch.tensor(fxfycxcy_list, dtype=torch.float32)  # (num_frames, 4)
        images = torch.stack(image_list, dim=0)
        c2ws = np.stack([np.array(frame["w2c"]) for frame in frames])
        c2ws = np.linalg.inv(c2ws)  # (num_frames, 4, 4)
        c2ws = torch.from_numpy(c2ws).float()
        
        c2w_bucket = repeat(torch.eye(4, dtype=torch.float32), 'h w -> b h w', b=c2ws.shape[0]).clone()
        c2w_bucket[:, :3] = c2ws[:, :3]  # (num_frames, 4, 4)

        random_crop_ratio = np.random.uniform(self.random_crop, 1.0) if random_crop_ratio is None else random_crop_ratio
        if random_crop_ratio < 1.0:
            magnify_ratio = 1.0 / random_crop_ratio
            cur_h, cur_w = images.shape[2], images.shape[3]
            images = F.interpolate(images, scale_factor=magnify_ratio, mode='bilinear', align_corners=False)
            mag_h, mag_w = images.shape[2], images.shape[3]
            start_h = (mag_h - cur_h) // 2
            start_w = (mag_w - cur_w) // 2
            images = images[:, :, start_h:start_h+cur_h, start_w:start_w+cur_w]
            intrinsics[:, 0] *= (mag_w / cur_w)
            intrinsics[:, 1] *= (mag_h / cur_h)

        return images, intrinsics, c2w_bucket, random_crop_ratio

    def __getitem__(self, idx):
        try:
            with open(self.json_path, "r") as f:
                eval_data = json.load(f)
            scene_dict = {item["scene_name"]: item for item in eval_data}
            data_path = self.data_path[idx]
            data_json = json.load(open(data_path, 'r'))
            scene_name = data_json['scene_name']
            frames = data_json['frames']
            image_base_dir = data_path.rsplit('/', 1)[0]
     
            # read config
            input_frame_select_type = self.config.data.input_frame_select_type
            target_frame_select_type = self.config.data.target_frame_select_type
            num_input_frames = self.config.data.num_input_frames
            num_target_frames = self.config.data.get("num_target_frames", 0)
            if num_target_frames == 0:
                assert target_frame_select_type == 'uniform_every'
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
                assert min_frame_dist >= max(num_input_frames, num_target_frames) - 1
            else:
                assert min_frame_dist >= num_input_frames + num_target_frames - 1
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
                target_frame_idx = np.random.choice(frame_idx, num_target_frames, replace=False)
            elif target_frame_select_type == 'uniform':
                target_frame_idx = np.linspace(start_frame_idx, end_frame_idx, num_target_frames, dtype=int)
            elif target_frame_select_type == 'uniform_every':
                uniform_every = self.config.data.target_uniform_every
                target_frame_idx = list(range(start_frame_idx, end_frame_idx + 1, uniform_every))
                num_target_frames = len(target_frame_idx)
            else:
                raise NotImplementedError
            target_frame_idx = sorted(target_frame_idx)
     
            # get input frames
            if not target_has_input:
                frame_idx = [x for x in frame_idx if x not in target_frame_idx]
            if input_frame_select_type == 'random':
                input_frame_idx = np.random.choice(frame_idx, num_input_frames, replace=False)
            elif input_frame_select_type == 'uniform':
                input_frame_idx = np.linspace(0, len(frame_idx) - 1, num_input_frames, dtype=int)
                input_frame_idx = [frame_idx[i] for i in input_frame_idx]
            elif input_frame_select_type == 'kmeans':
                input_frame_idx = scene_dict[scene_name][f"fold_8_kmeans_{self.config.data.num_input_frames}_input"]
            input_frame_idx = sorted(input_frame_idx)
            if reverse_input:
                input_frame_idx = input_frame_idx[::-1]
            if shuffle_input:
                np.random.shuffle(input_frame_idx)
     
            random_crop_ratio = None
            target_frames = [frames[i] for i in target_frame_idx]
            target_images, target_intr, target_c2ws, random_crop_ratio = self.process_frames(target_frames, image_base_dir)
     
            input_frames = [frames[i] for i in input_frame_idx]
            input_images, input_intr, input_c2ws, _ = self.process_frames(input_frames, image_base_dir, random_crop_ratio)

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
            image_indices = input_frame_idx + target_frame_idx
            image_indices = torch.tensor(image_indices).long().unsqueeze(-1)

            ret_dict = {
                "image": image,  # (num_input + num_target, 3, resize_h, resize_w)
                "fxfycxcy": fxfycxcy,  # (num_input + num_target, 4)
                "c2w": c2w,  # (num_input + num_target, 4, 4)
                "index": image_indices,
                "scene_name": scene_name,
                # "input_frame_idx": torch.tensor(input_frame_idx).long(),
                # "target_frame_idx": torch.tensor(target_frame_idx).long(),
            }

        except:
            traceback.print_exc()
            print(f"error loading data: {self.data_path[idx]}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        return ret_dict
