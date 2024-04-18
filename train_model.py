import torch
import matplotlib.pyplot as plt
from plots import plot_side_by_side_raster


def train_discriminator(
    generator,
    discriminator,
    discriminator_optimizer,
    bce_loss_function,
    low_res,
    high_res,
    device,
    freeze=False,
):
    if freeze:
        # Freeze the weights of the discriminator
        for param in discriminator.parameters():
            param.requires_grad = False
    else:
        # Unfreeze the weights of the discriminator
        for param in discriminator.parameters():
            param.requires_grad = True

    high_res_generator = generator(low_res)

    high_res_disc_real = discriminator(high_res)
    high_res_disc_fake = discriminator(high_res_generator.detach())

    # discriminator loss
    real_pred_loss = bce_loss_function(
        high_res_disc_real, torch.ones_like(high_res_disc_real).to(device)
    )

    fake_pred_loss = bce_loss_function(
        high_res_disc_fake, torch.zeros_like(high_res_disc_fake).to(device)
    )
    discriminator_loss = real_pred_loss + fake_pred_loss

    # get loss
    d_loss = discriminator_loss.item()*low_res.shape[0]
    # optimize only if training
    if not freeze:
        # reset the discriminator gradient
        discriminator_optimizer.zero_grad()
        # backpropagation
        discriminator_loss.backward()
        # update the model parameters
        discriminator_optimizer.step()

    return high_res_generator, d_loss


def train_generator(
    generator,
    discriminator,
    generator_optimizer,
    bce_loss_function,
    mse_loss_function,
    high_res,
    high_res_generator,
    alpha,
    device,
    freeze=False,
):
    if freeze:
        # Freeze the weights of the generator
        for param in generator.parameters():
            param.requires_grad = False
    else:
        # Unfreeze the weights of the generator
        for param in discriminator.parameters():
            param.requires_grad = True

    high_res_disc_fake = discriminator(high_res_generator)

    # generator loss
    content_loss = mse_loss_function(high_res, high_res_generator)
    adversarial_loss = bce_loss_function(
        high_res_disc_fake, torch.ones_like(high_res_disc_fake).to(device)
    )
    generator_loss = content_loss + alpha * adversarial_loss

    # get loss
    g_loss = generator_loss.item()*high_res.shape[0]

    # optimize only if training
    if not freeze:
        # reset the generator gradient
        generator_optimizer.zero_grad()
        # backpropagation
        generator_loss.backward()
        # update the model parameters
        generator_optimizer.step()

    return g_loss


def SRGAN_training(
    generator,
    discriminator,
    train_dataloader,
    saving_path_srgan,
    alpha_=1e-3,
    G_learning_rate=1e-5,
    D_learning_rate=1e-5,
    number_of_epochs=100,
    device="cpu",
    balanced_training=False,
    resume_training=False,
    pre_training=False,
    pre_train_number_of_epochs=10,
    variable_name="swvl1",
):
    # optimizers
    generator_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, generator.parameters()),
        lr=G_learning_rate,
        betas=(0.9, 0.999),
    )
    discriminator_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, discriminator.parameters()),
        lr=D_learning_rate,
        betas=(0.9, 0.999),
    )

    bce_loss_function = torch.nn.BCELoss()
    mse_loss_function = torch.nn.MSELoss()

    # if pre_training:
    #     alpha = 0

    if resume_training:
        checkpoint = torch.load(saving_path_srgan, map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        generator_optimizer.load_state_dict(checkpoint["G_state_dict"])
        discriminator_optimizer.load_state_dict(checkpoint["D_state_dict"])
        G_Loss_list = checkpoint["generator_loss"]
        D_Loss_list = checkpoint["discriminator_loss"]
        start_epoch = checkpoint["epoch"] + 1
    else:
        G_Loss_list = []
        D_Loss_list = []
        start_epoch = 0

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    for epoch in range(start_epoch, number_of_epochs):
        # check if pre_training or not
        if pre_training and (start_epoch < pre_train_number_of_epochs):
            pre_training = pre_training
            alpha = 0
        else:
            pre_training = False
            alpha = alpha_
        # Losses initialization
        G_loss = 0.0
        D_loss = 0.0
        freeze_generator = False
        freeze_discriminator = pre_training

        for i, batch in enumerate(train_dataloader):
            low_res, high_res = batch
            low_res, high_res = low_res.to(device), high_res.to(device)
            generator.train()
            discriminator.train()

            # -----------------------------Train discriminator-----------------
            high_res_generator, d_loss = train_discriminator(
                generator,
                discriminator,
                discriminator_optimizer,
                bce_loss_function,
                low_res,
                high_res,
                device,
                freeze_discriminator,
            )
            D_loss += d_loss
            # -----------------------------Train generator---------------------

            g_loss = train_generator(
                generator,
                discriminator,
                generator_optimizer,
                bce_loss_function,
                mse_loss_function,
                high_res,
                high_res_generator,
                alpha,
                device,
                freeze_generator,
            )
            G_loss += g_loss

        # log the metrics, images, etc
        G_loss /= len(train_dataloader)
        D_loss /= len(train_dataloader)
        G_Loss_list.append(G_loss)
        D_Loss_list.append(D_loss)

        if epoch % 10 == 0:
            lr_image = low_res[0, ...].cpu().squeeze().detach().numpy()
            hr_image = high_res[0, ...].cpu().squeeze().detach().numpy()
            hr_image_model = high_res_generator[0, ...].cpu().squeeze().detach().numpy()

            rasters = {
                "ERA5 Interpolated (low resolution)": lr_image,
                "ERA5Land (high resolution)": hr_image,
                "ERA5Land by Model (high resolution)": hr_image_model,
            }
            plot_side_by_side_raster(rasters, variable_name, 6)
            plt.show()

        if balanced_training:
            if D_loss > 0.6:
                # train only discriminator next step
                freeze_generator = True
            elif D_loss < 0.45:
                # train only generator
                freeze_discriminator = True

            else:
                freeze_generator = False
                freeze_discriminator = False

        print(
            "epoch : ",
            epoch,
            " freeze generator : ",
            freeze_generator,
            " freeze discriminator : ",
            freeze_discriminator,
            "-----------> generator loss : ",
            G_loss,
            " discriminator loss : ",
            D_loss,
        )

        # saving state
        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "G_state_dict": generator_optimizer.state_dict(),
                "D_state_dict": discriminator_optimizer.state_dict(),
                "generator_loss": G_Loss_list,
                "discriminator_loss": D_Loss_list,
            },
            saving_path_srgan,
        )
    # Metrics plot

    plt.plot(G_Loss_list)
    plt.title("Generator Training Loss")
    plt.figure()
    plt.plot(D_Loss_list)
    plt.title("Discriminator Training Loss")

    plt.show()
