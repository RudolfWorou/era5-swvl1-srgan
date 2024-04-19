import streamlit as st
import matplotlib.pyplot as plt
import base64
import torch
from downscale_era5_swvl1 import super_resolve_swvl1_world

from rasters_manipulation import load_and_clean_raster
from models import Generator
from pathlib import Path


def main():
    st.title("ERA5-SWVL1-SR")

    st.markdown(
        """
    ERA5-SWVL1-SR is a generative model, which performs super resolution on ERA5 swvl1 variable (soil moisture in the top $7cm$ soil layer), and outputs ERA5-Land swvl1 variable.
         
    More information about the model and its architecture is available on [Github](https://github.com/RudolfWorou/era5-swvl1-srgan).
       
    In the current version of the space, only **one timestamp** can be super-resolved. Make sure the ERA5 file uploaded is in **netcdf** format and only contains one timestamp.
    The ERA5-Land output data can be downloaded in a netcdf format. 
        """
    )
    # Sidebar for user input
    st.subheader("Super resolution Settings")
    st.markdown(
        """Overalapping patches are used to have a better output by averaging multiple blocks."""
    )
    uploaded_file = st.file_uploader("Upload ERA5 .nc file", type=["nc"])
    overlapping_patches = st.select_slider(
        "Overlapping patches",
        options=[1, 2, 4, 6, 8],
        value=2,
    )

    if uploaded_file is not None:
        # Load and plot the input raster
        era5_raster = load_and_clean_raster(
            uploaded_file,
            tolerance=1e-6,
        )

        st.subheader("ERA5 Input Raster")
        plot_raster(era5_raster)

        # get best model

        generator = Generator()
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        generator.load_state_dict(checkpoint["generator_state_dict"])

        with st.spinner("The model is running..."):
            era5land_model_raster = super_resolve_swvl1_world(
                era5_raster,
                era5_raster.time.values[0],
                generator,
                DEVICE,
                BATCH_SIZE,
                NUM_WORKERS,
                overlapping_patches=overlapping_patches,
            )

        # Display the super-resolved raster
        st.subheader("Generated ERA5-Land Raster")
        with st.spinner("Ploting the raster and generating the download link..."):
            plot_raster(era5land_model_raster)
            # Download result
            st.markdown(
                get_binary_file_downloader_html(era5land_model_raster),
                unsafe_allow_html=True,
            )


def plot_raster(raster):
    fig, ax = plt.subplots(figsize=(10, 5))
    raster.swvl1.plot(ax=ax)
    st.pyplot(fig)


def get_binary_file_downloader_html(data):
    # Function to create a download link for the super-resolved raster
    b64 = base64.b64encode(data.to_netcdf()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="era5-land_output_raster.nc">Download Generated ERA5-Land Raster</a>'
    return href


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    MODEL_PATH = Path("models/final_state/srgan_model.pth")
    main()
