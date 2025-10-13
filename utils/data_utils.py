from pathlib import Path
from typing import Any, Dict
from torch.utils.data import IterableDataset, DataLoader

from openvla.prismatic.vla.datasets.rlds import make_interleaved_dataset
from openvla.prismatic.vla.datasets.rlds.oxe import (
    OXE_NAMED_MIXTURES,
    get_oxe_dataset_kwargs_and_weights,
)
from openvla.prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

import numpy as np

# === Raw Batch Transform ===
class RLDSBatchTransform:
    def __call__(self, rlds_batch: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dataset_name": rlds_batch["dataset_name"],
            "image": np.asarray(rlds_batch["observation"]["image_primary"]),        
            "wrist_image": np.asarray(rlds_batch["observation"]["image_wrist"]),             
            "action": rlds_batch["action"],                                       
            "language_instruction": rlds_batch["task"]["language_instruction"].decode().lower()      
        }


# === RLDS Dataset (No Resize/Augmentation) ===
class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        data_mix: str,
        batch_transform: RLDSBatchTransform,
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
    ) -> None:
        self.data_root_dir = data_root_dir
        self.data_mix = data_mix
        self.batch_transform = batch_transform

        if self.data_mix in OXE_NAMED_MIXTURES:
            mixture_spec = OXE_NAMED_MIXTURES[self.data_mix]
        else:
            mixture_spec = [(self.data_mix, 1.0)]

        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            load_camera_views=("primary", "wrist"),
            load_depth=False,
            load_proprio=False,
            load_language=True,
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )

        rlds_config = dict(
            traj_transform_kwargs=dict(
                window_size=1,
                future_action_window_size=0,
                skip_unlabeled=False,
                goal_relabeling_strategy=None,
            ),
            frame_transform_kwargs=dict(),  # <<< no resize_size at all
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=True,
            traj_transform_threads=len(mixture_spec),
            traj_read_threads=len(mixture_spec),
            train=train,
        )

        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.take(self.dataset_length).as_numpy_iterator():
            yield self.batch_transform(rlds_batch)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not support indexing")


# === Identity Collator ===
def identity_collate_fn(batch):
    return dict(
        image=np.squeeze(np.stack([b["image"] for b in batch], axis=0)),
        wrist_image=np.squeeze(np.stack([b["wrist_image"] for b in batch], axis=0)),
        action=np.squeeze(np.stack([b["action"] for b in batch], axis=0)).shape,
        language_instruction=[b["language_instruction"] for b in batch],
    )


# === Final Dataloader Constructor ===
def RLDSDataLoader(cfg) -> DataLoader:
    """
    cfg must contain:
        - data_root_dir: Path
        - dataset_name: str
        - batch_size: int
        - shuffle_buffer_size: int
    """
    batch_transform = RLDSBatchTransform()

    vla_dataset = RLDSDataset(
        data_root_dir=cfg.data_root_dir,
        data_mix=cfg.dataset_name,
        batch_transform=batch_transform,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        train=True,
    )

    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=identity_collate_fn,
        num_workers=0,  # TFDS manages parallelism internally
    )

    return dataloader
