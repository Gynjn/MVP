import random
from data.batched_sampler import DynamicBatchedMultiFeatureRandomSampler
import numpy as np

class BaseDataset:
    def __init__(
        self,
        config,
    ):
        self.num_views = config.training.num_views
        self._set_resolutions(config.training.res_dict)
        self._set_input_views(config.training.num_views)
        self._set_target_views(config.training.target_views)
        print(self._input_views)
        print(self._target_views)
        print(self._resolutions)
        # Initialize the seed for the random number generator
        self.seed = config.data.seed
        self.max_num_retries = 10

    def set_epoch(self, epoch):
        """
        Set the current epoch for all constituent datasets.

        Args:
            epoch (int): The current epoch number
        """
        pass  # nothing to do by default

    def make_sampler(
        self,
        batch_size_per_gpu,
        shuffle: bool = True,
        world_size: int = 1,
        rank: int = 0,
        drop_last: bool = True,
        use_dynamic_sampler: bool = False,
    ):
        if not (shuffle):
            raise NotImplementedError("Only shuffle=True is supported for now.")
        
        num_of_num_views = len(self.num_views)
        feature_pool_sizes = [num_of_num_views]
        scaling_feature_idx = 0

        feature_to_batch_size_map = {
            i: bs for i, bs in enumerate(batch_size_per_gpu)
        }
        
        return DynamicBatchedMultiFeatureRandomSampler(
            dataset=self,
            pool_sizes=feature_pool_sizes,
            scaling_feature_idx=scaling_feature_idx,
            feature_to_batch_size_map=feature_to_batch_size_map,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
        )
            

    def _scene_len(self):
        self.data_path = []  # This should be set in the subclass
        self.num_of_scenes = len(self.data_path)

    def __len__(self):
        "Length of the dataset is determined by the number of scenes in the dataset split"
        return self.num_of_scenes

    def _get_views(self, idx, resolution, num_views_to_input, num_views_to_target):
        raise NotImplementedError()

    def _set_resolutions(self, resolutions):
        self._resolutions = {
            i: tuple(res) for i, res in enumerate(resolutions)
        }

    def _set_input_views(self, num_views):
        self._input_views = dict(enumerate(num_views))


    def _set_target_views(self, target_views):
        self._target_views = dict(enumerate(target_views))

    def _getitem_fn(self, idx):
        idx, dict_idx = idx
        resolution = self._resolutions[dict_idx]
        input_view = self._input_views[dict_idx]
        target_view = self._target_views[dict_idx]
        # print(resolution, input_view, target_view)

        views = self._get_views(idx, resolution, input_view, target_view)

        return views

    def __getitem__(self, idx):
        num_retries = 0
        while num_retries <= self.max_num_retries:
            try:
                return self._getitem_fn(idx)
            except Exception as e:
                num_retries += 1                
                if isinstance(idx, tuple):
                    # The scene index is the first element of the tuple
                    idx_list = list(idx)
                    idx_list[0] = np.random.randint(0, len(self))
                    idx = tuple(idx_list)
                else:
                    # The scene index is idx
                    idx = np.random.randint(0, len(self))

        # except:
            # self._getitem_fn(random.randint(0, len(self) - 1))