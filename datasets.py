from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
from rasters_manipulation import convert_era5_raster_to_tensor, cut_tensor_in_pieces


class SRDataset(Dataset):
    def __init__(
        self,
        lr_data_path,
        hr_data_path,
        upscale_factor=4,
        latitude_size=108,
        longitude_size=194,
        low_res_step=0.4,
        high_res_step=0.1,
        low_res_image_dim=32,
        high_res_image_dim=128,
    ):
        self.low_res_step = low_res_step
        self.high_res_step = high_res_step
        self.low_res_image_dim = low_res_image_dim
        self.high_res_image_dim = high_res_image_dim

        self.upscale_factor = upscale_factor

        self.latitude_size = latitude_size
        self.longitude_size = longitude_size

        self.era5_paths = list(lr_data_path.iterdir())
        self.era5_land_paths = list(hr_data_path.iterdir())

    def __len__(self):
        return len(self.era5_paths)

    def __getitem__(self, index):
        i = np.random.choice(range(self.latitude_size - self.low_res_image_dim))
        j = np.random.choice(range(self.longitude_size - self.low_res_image_dim))

        lr_im = torch.load(self.era5_paths[index])[
            i : i + self.low_res_image_dim, j : j + self.low_res_image_dim
        ].repeat(1, 1, 1)

        hr_im = torch.load(self.era5_land_paths[index])[
            i * self.upscale_factor : i * self.upscale_factor + self.high_res_image_dim,
            j * self.upscale_factor : j * self.upscale_factor + self.high_res_image_dim,
        ].repeat(1, 1, 1)

        return lr_im, hr_im


class SRRasterDataset(Dataset):
    def __init__(
        self,
        era5_data,
        time,
        offset_height,
        offset_width,
        variable="swvl1",
        low_res_step=0.4,
        piece_size=32,
    ):
        self.time = pd.Timestamp(time)
        self.input_tensor = convert_era5_raster_to_tensor(
            era5_data, variable, low_res_step
        )
        _, self.height, self.width = self.input_tensor.shape
        self.offset_height = offset_height
        self.offset_width = offset_width
        self.pieces = cut_tensor_in_pieces(
            self.input_tensor,
            piece_size,
            offset_height=offset_height,
            offset_width=offset_width,
        )

    def __len__(self):
        return len(self.pieces)

    def __getitem__(self, index):
        return index, self.pieces[index]
