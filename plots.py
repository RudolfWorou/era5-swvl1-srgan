import matplotlib.pyplot as plt
import numpy as np
import torch


def random_window_endpoints(array, k):
    if k > len(array):
        raise ValueError("Window length k cannot be greater than the array length")

    start_index = np.random.randint(0, len(array) - k + 1)
    end_index = start_index + k - 1
    start_element = array[start_index]
    end_element = array[end_index]
    return start_element, end_element


def plot_side_by_side_raster(
    rasters: dict[str, np.ndarray],
    variable_name: str = "swvl1",
    width: int = 6,
):
    # Calculate the minimum and maximum values across all datasets
    combined_min = min([raster.min() for raster in rasters.values()])
    combined_max = max([raster.max() for raster in rasters.values()])

    # Create subplots side by side
    fig, ax = plt.subplots(
        1,
        len(rasters),
        figsize=(
            width * len(rasters),
            width,
        ),
    )
    if len(rasters) == 1:
        ax = [ax]
    # Plot data
    img = {}
    for i, raster_id in enumerate(rasters):
        img[raster_id] = ax[i].imshow(
            rasters[raster_id],
            cmap="viridis",
            vmin=combined_min,
            vmax=combined_max,
        )
        ax[i].set_title(raster_id)
        ax[i].grid(False)

        # Add coastlines and gridlines
        # ax[i].coastlines()
        # ax[i].gridlines()

    # Adjust the position of the colorbar to the side
    cax = fig.add_axes([1.0, 0.0, 0.02, 1.0])
    fig.colorbar(
        img[raster_id],
        cax=cax,
        orientation="vertical",
        label=variable_name,
    )


def image_look_evaluation(
    dataloader,
    model,
    nb_samples,
    variable_name: str = "swvl1",
    width: int = 6,
    device="cuda",
    seed=42,
):
    model = model.to(device)
    np.random.seed(seed)
    random_indices = np.random.choice(
        len(dataloader.dataset), size=nb_samples, replace=False
    )
    random_data = [dataloader.dataset[i] for i in random_indices]

    with torch.no_grad():
        for batch in random_data:
            low_res, high_res = batch
            low_res, high_res = low_res.to(device), high_res.to(device)

            high_res_prediction_1 = model(low_res)

            lr_image = low_res[0, ...].cpu().squeeze().numpy()
            hr_image = high_res[0, ...].cpu().squeeze().numpy()
            hr_image_model = high_res_prediction_1[0, ...].cpu().squeeze().numpy()

            rasters = {
                "ERA5 Interpolated (low resolution)": lr_image,
                "ERA5Land (high resolution)": hr_image,
                "ERA5Land by Model (high resolution)": hr_image_model,
            }
            plot_side_by_side_raster(rasters, variable_name, width)
