import os
import shutil
from datetime import datetime


def create_model_folder_and_copy_json(
    model_path_parent,
    model_name="srgan_model.pth",
    resume_training=True,
    train_model=True,
    model_folder_path=None,
    source_json_path=None,
):
    if train_model and not resume_training:
        current_datetime = datetime.now()
        model_folder_path = model_path_parent / current_datetime.strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        os.makedirs(model_folder_path)
        destination_json_path = model_folder_path / source_json_path.name
        shutil.copy(source_json_path, destination_json_path)
        model_path = model_folder_path / model_name

    else:
        (model_path, destination_json_path) = get_model_folder_and_json(
            model_folder_path
        )
    return model_path, destination_json_path


def get_model_folder_and_json(model_folder_path):
    files_paths = list(model_folder_path.iterdir())
    assert len(files_paths) == 2
    for file_path in files_paths:
        if file_path.suffix == ".pth":
            model_path = file_path
        else:
            destination_json_path = file_path
    return model_path, destination_json_path
