import torch 
from torch.utils.data import Dataset
import numpy as np


class MaskedGAFDataset(Dataset):
    def __init__(self, x_array, patch_size=4, mask_ratio=0.4, mask_value=0.0):
        assert x_array.ndim == 4, f"Expected (N, C, H, W), got {x_array.shape}"
        assert x_array.shape[1] == 1, "Expected single-channel GAF input"
        assert x_array.shape[2] % patch_size == 0
        assert x_array.shape[3] % patch_size == 0

        self.x = x_array.astype(np.float32)
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mask_value = mask_value

        _, _, self.H, self.W = self.x.shape
        self.num_patch_rows = self.H // patch_size
        self.num_patch_cols = self.W // patch_size
        self.num_patches = self.num_patch_rows * self.num_patch_cols

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        original = self.x[idx].copy()
        masked = original.copy()

        patch_mask = np.zeros(self.num_patches, dtype=np.float32)
        num_to_mask = max(1, int(self.num_patches * self.mask_ratio))
        masked_patch_indices = np.random.choice(self.num_patches, size=num_to_mask, replace=False)
        patch_mask[masked_patch_indices] = 1.0

        pixel_mask = np.zeros((1, self.H, self.W), dtype=np.float32)

        for patch_idx in masked_patch_indices:
            r = patch_idx // self.num_patch_cols
            c = patch_idx % self.num_patch_cols

            r0 = r * self.patch_size
            r1 = r0 + self.patch_size
            c0 = c * self.patch_size
            c1 = c0 + self.patch_size

            masked[:, r0:r1, c0:c1] = self.mask_value
            pixel_mask[:, r0:r1, c0:c1] = 1.0

        return {
            "masked_input": torch.from_numpy(masked),
            "target": torch.from_numpy(original),
            "mask": torch.from_numpy(pixel_mask)
        }
    
class GAFClassificationDataset(Dataset):
    def __init__(self, x_array, y_array):
        assert x_array.ndim == 4
        assert len(x_array) == len(y_array)
        self.x = x_array.astype(np.float32)
        self.y = y_array.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.tensor(self.y[idx], dtype=torch.long)
        )

# Weather fusion classifier
class GAFWeatherClassificationDataset(Dataset):
    def __init__(self, x_array, w_array, y_array):
        assert len(x_array) == len(w_array) == len(y_array)
        self.x = x_array.astype(np.float32)
        self.w = w_array.astype(np.float32)
        self.y = y_array.astype(np.int64)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.w[idx]),
            torch.tensor(self.y[idx], dtype=torch.long)
        )
