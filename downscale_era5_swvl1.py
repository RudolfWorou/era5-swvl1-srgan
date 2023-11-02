import torch
import numpy as np
from rasters_manipulation import (
    merge_pieces,
    create_raster_from_array,
    interpolate_era5_raster,
)
from datasets import SRRasterDataset
from plots import plot_side_by_side_raster


def downscale_era5_raster(
    sr_raster_dataset, ocean_mask, model, device, batch_size, num_workers=-1
):
    eval_r_ds = torch.utils.data.dataloader.DataLoader(
        sr_raster_dataset, batch_size=batch_size, num_workers=num_workers
    )

    hr_pieces = [None] * len(sr_raster_dataset)

    model = model.to(device)
    with torch.no_grad():
        for _, batch in enumerate(eval_r_ds):
            indices, low_res = batch
            indices = np.array(indices)
            low_res = low_res.to(device)

            high_res_prediction_1 = model(low_res)

            for i, index in enumerate(indices):
                hr_pieces[index] = high_res_prediction_1[i].to("cpu")

    reconstructed_tensor = merge_pieces(
        hr_pieces,
        sr_raster_dataset.height,
        sr_raster_dataset.width,
        sr_raster_dataset.offset_height,
        sr_raster_dataset.offset_width,
    )

    return reconstructed_tensor


def super_resolve_swvl1_world(
    era5_raster,
    time,
    model,
    device,
    batch_size,
    num_workers=-1,
    piece_size=16,
    overlapping_patches=8,
    ocean_mask_path="data/ocean_mask.npy",
):
    reconstructed_tensors = []
    for offset in [0, 2, 4, 6, 8, 10, 12, 14][:overlapping_patches]:
        era5land_raster_dataset = SRRasterDataset(
            era5_raster,
            time=time,
            piece_size=piece_size,
            offset_height=offset,
            offset_width=offset,
        )
        ocean_mask = np.load(ocean_mask_path)
        era5land_model_tensor = downscale_era5_raster(
            era5land_raster_dataset,
            ocean_mask,
            model,
            device,
            batch_size,
            num_workers,
        )
        reconstructed_tensors.append(era5land_model_tensor)

    reconstructed_tensor = torch.nanmean(
        torch.stack(reconstructed_tensors),
        dim=0,
    )
    raster = create_raster_from_array(
        reconstructed_tensor.numpy().squeeze(),
        latitude=90,
        longitude=0,
        time=era5land_raster_dataset.time,
        attributes={
            "Conventions:": "CF-1.6",
            "history": "Downscaled with ERA5-SWVL1-SR",
        },
        latitude_step=0.1,
        longitude_step=0.1,
    )

    raster = raster.where(ocean_mask, np.nan)

    return raster


def super_resolve_swvl1_local_patch(
    model,
    era5_raster,
    latitude,
    longitude,
    era5land_raster=None,
    patch_size=16,
    upscale_factor=4,
    device="cpu",
    verbose=True,
):
    model = model.to(device)

    era5_patch_size = int(patch_size * (0.4 / 0.25))
    era5_array = era5_raster.sel(
        latitude=slice(latitude, None), longitude=slice(longitude, None)
    ).swvl1.values[:, :era5_patch_size, :era5_patch_size]

    era5_interpolated_raster = interpolate_era5_raster(era5_raster)
    input_array = era5_interpolated_raster.sel(
        latitude=slice(latitude, None), longitude=slice(longitude, None)
    ).swvl1.values[:, :patch_size, :patch_size]

    model_output_array = (
        model(torch.tensor(input_array, dtype=torch.float32).to(device))
        .detach()
        .to("cpu")
        .numpy()
    )

    plot_dict = {
        "ERA5": era5_array.squeeze(),
        "ERA5 Interpolated": input_array.squeeze(),
        "ERA5Land from the model": model_output_array.squeeze(),
    }

    if era5land_raster is not None:
        output_patch_size = patch_size * upscale_factor
        output_array = era5land_raster.sel(
            latitude=slice(latitude, None), longitude=slice(longitude, None)
        ).swvl1.values[:, :output_patch_size, :output_patch_size]
        plot_dict["ERA5Land"] = np.nan_to_num(output_array, nan=0).squeeze()

    if verbose:
        plot_side_by_side_raster(
            plot_dict,
            width=6,
        )
    return model_output_array
