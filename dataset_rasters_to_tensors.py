import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

from rasters_manipulation import load_and_clean_raster


def export_era5_data_to_tensors(
    data_path=Path("bucket/test/lr"),
    save_path=Path("bucket_tensor/test/lr"),
    variable="swvl1",
    low_res_step=0.4,
    tolerance=1e-6,
):
    for file_path in tqdm(data_path.iterdir()):
        era5_data = load_and_clean_raster(file_path, tolerance)

        min_latitude = round(era5_data.latitude.values.min(), 2)
        max_latitude = round(era5_data.latitude.values.max(), 2)

        min_longitude = round(era5_data.longitude.values.min(), 2)
        max_longitude = round(era5_data.longitude.values.max(), 2)

        for time in tqdm(era5_data.time.values):
            file_name = time.astype("datetime64[D]").astype(str)
            era5_day_data = era5_data.sel(time=time)
            era5_day_data = era5_day_data.interp(
                longitude=(
                    np.arange(
                        0, int((max_longitude - min_longitude) / low_res_step) + 1
                    )
                    * low_res_step
                    + min_longitude
                ).round(2),
                latitude=(
                    np.arange(0, int((max_latitude - min_latitude) / low_res_step) + 1)
                    * (-low_res_step)
                    + max_latitude
                ).round(2),
                method="cubic",
                kwargs={"fill_value": "extrapolate"},
            )
            era5_day_data = era5_day_data[["latitude", "longitude", "time", variable]]

            torch.save(
                torch.tensor(era5_day_data[variable].to_numpy(), dtype=torch.float32),
                save_path / f"{file_name}.pth",
            )


def export_era5_land_data_to_tensors(
    data_path=Path("bucket/test/lr"),
    save_path=Path("bucket_tensor/test/lr"),
    variable="swvl1",
    tolerance=1e-6,
):
    # save_path /
    for file_path in tqdm(data_path.iterdir()):
        era5_land_data = load_and_clean_raster(file_path, tolerance)
        for time in tqdm(era5_land_data.time.values):
            file_name = time.astype("datetime64[D]").astype(str)
            era5_land_day_data = era5_land_data.sel(time=time)

            era5_land_day_data = era5_land_day_data[
                ["latitude", "longitude", "time", variable]
            ]

            torch.save(
                torch.tensor(
                    era5_land_day_data[variable].to_numpy(), dtype=torch.float32
                ),
                save_path / f"{file_name}.pth",
            )


if __name__ == "__main__":
    bucket_path = Path("bucket")
    bucket_tensor_path = Path("bucket_tensor")

    for data_type in ["train", "test"]:
        export_era5_data_to_tensors(
            data_path=bucket_path / data_type / "era5",
            save_path=bucket_tensor_path / data_type / "era5",
        )

        export_era5_land_data_to_tensors(
            data_path=bucket_path / data_type / "era5land",
            save_path=bucket_tensor_path / data_type / "era5land",
        )
