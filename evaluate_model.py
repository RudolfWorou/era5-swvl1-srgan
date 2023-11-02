import pandas as pd
from pytorch_msssim import ssim
import torch


def PSNR(image_1, image_2):
    return 10 * torch.log10(1.0 / torch.nn.MSELoss()(image_1, image_2))


def evaluate_model(model, dataloader, device):
    model = model.to(device)

    test_frame = pd.DataFrame(
        columns=["PSNR", "SSIM", "Average(PSNR,SSIM)", "MSE", "MAE"]
    )

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            low_res, high_res = batch
            low_res, high_res = low_res.to(device), high_res.to(device)
            # forward pass through the model
            high_res_prediction = model(low_res)
            PSNR_loss = PSNR(high_res_prediction, high_res).item()
            SSIM_loss = ssim(high_res_prediction, high_res, data_range=1).item()
            mse_loss = torch.nn.MSELoss()(high_res_prediction, high_res).item()
            mae_loss = torch.nn.L1Loss()(high_res_prediction, high_res).item()
            n = len(test_frame)
            test_frame.loc[n] = [
                PSNR_loss,
                SSIM_loss,
                0.5 * (PSNR_loss + SSIM_loss),
                mse_loss,
                mae_loss,
            ]

    return test_frame


def evaluate_model_robustness(model, dataloader, device, noise_power=0.05):
    model = model.to(device)

    test_frame = pd.DataFrame(columns=["MEAN_lr", "VAR_lr", "MEAN_hr", "VAR_hr"])

    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            low_res, high_res = batch
            low_res, high_res = low_res.to(device), high_res.to(device)

            low_res_variance = torch.var(low_res, dim=(1, 2, 3), keepdim=True)
            noisy_low_res = (
                low_res + torch.randn_like(low_res) * noise_power * low_res_variance
            )

            # forward pass through the model
            high_res_prediction = model(low_res)
            noisy_high_res_prediction = model(noisy_low_res)

            diff_mse_lr = torch.mean(noisy_low_res - low_res).item()

            diff_var_lr = torch.var(noisy_low_res - low_res).item()

            diff_mse_hr = torch.mean(
                noisy_high_res_prediction - high_res_prediction
            ).item()

            diff_var_hr = torch.var(
                noisy_high_res_prediction - high_res_prediction
            ).item()

            n = len(test_frame)
            test_frame.loc[n] = [diff_mse_lr, diff_var_lr, diff_mse_hr, diff_var_hr]

    return test_frame
