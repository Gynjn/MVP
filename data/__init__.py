import torch

from data.dataset import Dataset
import torch.distributed as dist

class DynamicBatchDatasetWrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, batch_indices):
        """
        Handle batch of indices from DynamicBatchedMultiFeatureRandomSampler.

        Args:
            batch_indices: List of tuples like [(sample_idx, feat_idx_1, feat_idx_2, ...), ...]

        Returns:
            List of samples from the underlying dataset
        """
        if isinstance(batch_indices, (list, tuple)) and len(batch_indices) > 0:
            # If it's a batch (list of tuples), process each item
            if isinstance(batch_indices[0], (list, tuple)):
                return [self.dataset[idx] for idx in batch_indices]
            else:
                # Single tuple, call dataset directly
                return self.dataset[batch_indices]
        else:
            # Fallback for single index
            return self.dataset[batch_indices]

    def __len__(self):
        return len(self.dataset)
    
def get_train_data_loader(
    config,
    num_workers,
    shuffle=True,
    drop_last=True,
    pin_mem=True,
):
    
    dataset = Dataset(config)
    
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0

    batch_sampler = dataset.make_sampler(
        batch_size_per_gpu=config.training.batch_size_per_gpu,
        shuffle=shuffle,
        world_size=world_size,
        rank=rank,
        drop_last=drop_last,
        use_dynamic_sampler=True,
    )

    warpped_dataset = DynamicBatchDatasetWrapper(dataset)

    data_loader = torch.utils.data.DataLoader(
        warpped_dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return data_loader