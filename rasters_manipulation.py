import operator
import pandas as pd
import numpy as np
import xarray as xr
import torch


def slice_raster(
    raster,
    min_latitude,
    max_latitude,
    min_longitude,
    max_longitude,
):
    return raster.sel(
        latitude=slice(min_latitude, max_latitude),
        longitude=slice(min_longitude, max_longitude),
    )


def load_and_clean_raster(file_path, tolerance):
    raster = xr.open_dataset(file_path)

    raster = raster.fillna(0.0)
    raster = raster.where(raster > tolerance, other=0.0)

    raster = raster.sortby("latitude", ascending=False)
    raster = raster.sortby("longitude", ascending=True)

    return raster


def create_raster_from_array(
    data_array: np.ndarray,
    latitude: float,
    longitude: float,
    time: pd.Timestamp,
    attributes: dict,
    latitude_variation=operator.sub,
    longitude_variation=operator.add,
    latitude_step=0.25,
    longitude_step=0.25,
):
    latitude_values = latitude_variation(
        latitude, np.arange(data_array.shape[0]) * latitude_step
    )
    longitude_values = longitude_variation(
        longitude, np.arange(data_array.shape[1]) * longitude_step
    )

    # Create the xarray Dataset
    raster = xr.Dataset(
        dict(
            swvl1=(
                ["time", "latitude", "longitude"],
                np.expand_dims(data_array, axis=0),
            )
        ),
        coords={
            "longitude": longitude_values,
            "latitude": latitude_values,
            "time": [time],
        },
        attrs=attributes,
    )

    return raster


def interpolate_era5_raster(
    era5_day_data,
    variable="swvl1",
    low_res_step=0.4,
):
    min_latitude = round(era5_day_data.latitude.values.min(), 2)
    max_latitude = round(era5_day_data.latitude.values.max(), 2)

    min_longitude = round(era5_day_data.longitude.values.min(), 2)
    max_longitude = round(era5_day_data.longitude.values.max(), 2)

    era5_day_data = era5_day_data.interp(
        longitude=(
            np.arange(0, int((max_longitude - min_longitude) / low_res_step) + 1)
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
    return era5_day_data


def convert_era5_raster_to_tensor(
    era5_day_data,
    variable="swvl1",
    low_res_step=0.4,
):
    era5_day_data = interpolate_era5_raster(
        era5_day_data,
        variable,
        low_res_step,
    )
    return torch.tensor(
        era5_day_data[variable].to_numpy(),
        dtype=torch.float32,
    )


def cut_tensor_in_pieces(
    input_tensor,
    piece_size=32,
    default_value=0,
    offset_height=0,
    offset_width=0,
):
    _, height, width = input_tensor.shape
    pieces = []

    # Loop through the image and cut it into pieces
    for y in range(offset_height, height, piece_size):
        for x in range(offset_width, width, piece_size):
            piece = torch.ones((1, piece_size, piece_size)) * default_value
            current_piece = input_tensor[:, y : y + piece_size, x : x + piece_size]
            piece[:, : current_piece.shape[1], : current_piece.shape[2]] = current_piece

            pieces.append(piece)

    return pieces


def merge_pieces(
    pieces,
    height,
    width,
    offset_height,
    offset_width,
    factor=4,
    piece_size=16,
    hr_piece_size=64,
    max_height=1801,
    max_width=3600,
):
    # reconstructed_tensor = torch.zeros((1, height * factor, width * factor))
    reconstructed_tensor = torch.full(
        (1, height * factor, width * factor), float("nan")
    )
    _, hr_height, hr_width = reconstructed_tensor.shape
    # Counter for iterating through the pieces list
    piece_count = 0

    # Loop through the canvas and place each piece in its position
    for y in range(offset_height, height, piece_size):
        for x in range(offset_width, width, piece_size):
            y_limit = min(y * factor + hr_piece_size, hr_height - 1)
            x_limit = min(x * factor + hr_piece_size, hr_width - 1)

            reconstructed_tensor[
                :, y * factor : y_limit, x * factor : x_limit
            ] = pieces[piece_count][
                :, : (y_limit - y * factor), : (x_limit - x * factor)
            ]
            piece_count += 1

    return reconstructed_tensor[:, :max_height, :max_width]
