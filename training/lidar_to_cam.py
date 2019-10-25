from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import optparse

from utils.data.Dataloader import lidar_camera_dataloader
from utils.core.config import config

from utils.core.config import load_config
from utils.helpers.helpers import save_vanilla_model
from utils.helpers.helpers import display_two_images


from Generator import Generator
from Discriminator import PixelDiscriminator


import torch.optim as optim
import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


torch.manual_seed(0)
parser = optparse.OptionParser()

parser.add_option('-c', '--config', dest="config",
                  help="load this config file", metavar="FILE")

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def train(dataloader, config, device):
    camera_gen_losses = []
    camera_disc_losses = []

    # camera_weights = torch.tensor([
    #     1.4859513, 3.9798364, 13.7709121, 483.6851552, 926.3148902]).to(device=device, dtype=torch.float)
    # log2
    # camera_weights = torch.tensor([
    #     1.0, 1.9927091, 3.7835522, 8.9179244, 9.8553588]).to(device=device, dtype=torch.float)
    # criterion = nn.CrossEntropyLoss( reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    # criterion = nn.BCELoss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    criterion_gan = nn.MSELoss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)
    criterion_pixel = nn.L1Loss(reduction=config.TRAIN.DISCRIMINATOR_CRITERION_REDUCTION)


    camera_gen = Generator(5, 5, config.NUM_GPUS).to(device)
    camera_disc = PixelDiscriminator(5, 5, config.NUM_GPUS).to(device)


    if (device.type == 'cuda') and (config.NUM_GPUS > 1):
        camera_gen = nn.DataParallel(camera_gen, list(range(config.NUM_GPUS)))
        camera_disc = nn.DataParallel(camera_disc, list(range(config.NUM_GPUS)))

    optimizer_camera_gen = optim.Adam(camera_gen.parameters(), lr=config.CAMERA_GENERATOR.BASE_LR,
                                      betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))
    camera_gen_scheduler = optim.lr_scheduler.StepLR(optimizer_camera_gen,
                                                     step_size=config.CAMERA_GENERATOR.STEP_SIZE,
                                                     gamma=config.CAMERA_GENERATOR.STEP_GAMMA)
    optimizer_camera_disc = optim.Adam(camera_disc.parameters(), lr=config.CAMERA_DISCRIMINATOR.BASE_LR,
                                       betas=(config.TRAIN.BETA1, config.TRAIN.BETA2))
    camera_disc_scheduler = optim.lr_scheduler.StepLR(optimizer_camera_disc,
                                                      step_size=config.CAMERA_DISCRIMINATOR.STEP_SIZE,
                                                      gamma=config.CAMERA_DISCRIMINATOR.STEP_GAMMA)
    test_lidar_path1 = "/home/fatih/Inputs/test/46cameraView_0000000000.npz"
    test_camera_path1 = "/home/fatih/Inputs/test/46segmented_0000000000.npz"

    test_lidar_path2 = "/home/fatih/Inputs/test/01cameraView_0000000000.npz"
    test_camera_path2 = "/home/fatih/Inputs/test/01segmented_0000000000.npz"

    test_lidar1 = torch.from_numpy(np.load(test_lidar_path1)["data"].reshape(1, 5, 375, 1242)).to(device=device,
                                                                                                  dtype=torch.float)
    test_camera1 = torch.from_numpy(np.load(test_camera_path1)["data"].reshape(1, 5, 375, 1242)).to(device=device,
                                                                                                    dtype=torch.float)



    test_lidar2 = torch.from_numpy(np.load(test_lidar_path2)["data"].reshape(1, 5, 375, 1242)).to(device=device,
                                                                                                  dtype=torch.float)
    test_camera2 = torch.from_numpy(np.load(test_camera_path2)["data"].reshape(1, 5, 375, 1242)).to(device=device,
                                                                                                    dtype=torch.float)
    # camera_gen_total_params = sum(p.numel() for p in camera_gen.parameters())
    # print("Camera Generator ", camera_gen_total_params)

    # camera_disch_total_params = sum(p.numel() for p in camera_disc.parameters())
    # print("Camera Discriminator ", camera_disch_total_params)

    example_camera_output = []

    # if config.TRAIN.START_EPOCH > 0:
    #     print("loading previous model")
    #     checkpoint = torch.load(config.TRAIN.LOAD_WEIGHTS)
    #     camera_gen.load_state_dict(checkpoint['camera_gen'])
    #     camera_disc.load_state_dict(checkpoint['camera_disc'])
    #     optimizer_camera_gen.load_state_dict(checkpoint['optimizer_camera_gen'])
    #     optimizer_camera_disc.load_state_dict(checkpoint['optimizer_camera_disc'])
    #     camera_gen.train()
    #     camera_disc.train()
    #     print("done")

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.MAX_EPOCH):
        for current_batch, data in enumerate(dataloader, 0):
            if len(dataloader) - current_batch< config.TRAIN.BATCH_SIZE:
                continue

            label_real = Variable(torch.cuda.FloatTensor(np.ones((config.TRAIN.BATCH_SIZE, 1,23,77))), requires_grad=False)
            label_fake = Variable(torch.cuda.FloatTensor(np.zeros((config.TRAIN.BATCH_SIZE, 1,23,77))), requires_grad=False)

            #display_two_images(data["camera_data"][0], data["lidar_data"][0])
            camera_sample = data["camera_data"].to(device = device, dtype=torch.float)
            lidar_sample = data["lidar_data"].to(device = device, dtype=torch.float)

            ################################################################################
            #                               Zero Gradients
            ################################################################################

            optimizer_camera_gen.zero_grad()
            optimizer_camera_disc.zero_grad()

            ###############################################################################
            #                          Camera Generator
            ###############################################################################

            camera_gen.zero_grad()

            generated_camera_sample = camera_gen(lidar_sample)

            camera_disc_on_generated = camera_disc(generated_camera_sample, lidar_sample)

            camera_gen_loss_disc = criterion_gan(camera_disc_on_generated, label_real)
            camera_gen_loss_pixel =criterion_pixel(generated_camera_sample, camera_sample)
            camera_gen_loss_pixel = config.CAMERA_GENERATOR.PIXEL_LAMBDA * camera_gen_loss_pixel
            ################################################################################
            #                                   TEST LOSS START
            ###############################################################################
            lidar_mask = lidar_sample * generated_camera_sample.detach()
            camera_lidar_point_loss = criterion_pixel(lidar_mask, generated_camera_sample)
            camera_lidar_point_loss = config.CAMERA_GENERATOR.NEW_LOSS_LAMBDA * camera_lidar_point_loss
            camera_gen_loss_pixel = camera_gen_loss_pixel + camera_lidar_point_loss
            ################################################################################
            #                                   TEST LOSS END
            ###############################################################################
            camera_gen_loss = camera_gen_loss_disc + camera_gen_loss_pixel
            camera_gen_loss.backward()
            optimizer_camera_gen.step()

            ################################################################################
            #                           Camera Discriminator
            ################################################################################
            camera_disc.zero_grad()
            camera_disc_real_output = camera_disc(camera_sample, lidar_sample)
            camera_disc_real_loss = criterion_gan(camera_disc_real_output, label_real)

            camera_disc_fake_output= camera_disc(generated_camera_sample.detach(), lidar_sample)
            camera_disc_fake_loss = criterion_gan(camera_disc_fake_output, label_fake)
            camera_disc_loss = (camera_disc_fake_loss + camera_disc_real_loss) * 0.5

            camera_disc_loss.backward()
            optimizer_camera_disc.step()

            if current_batch % 5 == 0:
                print(
                    '[%d/%d][%d/%d]\t\t Lidar to Camera GAN Loss_D \t Real: %.4f  Fake: %.4f  Tot: %.4f \t'
                    'Loss_G \t GAN: %.4f  Pixel: %.4f  Tot: %.4f \t D(x): %.4f \t D(G(z)): %.4f  '
                    % (epoch, config.TRAIN.MAX_EPOCH, current_batch, len(dataloader),
                       camera_disc_real_loss.item(), camera_disc_fake_loss.item(), camera_disc_loss.item(),
                       camera_gen_loss_disc.item(), camera_gen_loss_pixel.item(), camera_gen_loss.item(),
                       camera_disc_real_output.mean().item(), camera_disc_fake_output.mean().item()))

            camera_gen_losses.append(camera_gen_loss.item())
            camera_disc_losses.append(camera_disc_loss.item())



            del generated_camera_sample
            del camera_sample, lidar_sample
            del label_real, label_fake


        camera_gen_scheduler.step()
        camera_disc_scheduler.step()

        fig = plt.figure(num=None, figsize=(25, 12), dpi=100, facecolor='w', edgecolor='k')
        plt.subplot(1, 2, 1)
        plt.title("Discriminator  Loss  Training")
        plt.plot(running_mean(camera_disc_losses, 100), label="Discriminator Loss")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title("Generator Loss Training")
        plt.plot(running_mean(camera_gen_losses, 100), label="Generator Loss")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()


        plt.savefig(config.TRAIN.GRAPH_SAVE_PATH+str(epoch))
        plt.close()



    #save_model(config, lidar_gen, camera_gen , lidar_disc , camera_disc, optimizer_lidar_gen,
    #           optimizer_camera_gen, optimizer_lidar_disc, optimizer_camera_disc )

        if epoch != 0:
            with torch.no_grad():
                fakeCamera1 = camera_gen(test_lidar1.detach())
                np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_generated_camera_1",
                                    data=fakeCamera1[-1].cpu().numpy())
                np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_lidar_1",
                                    data=test_lidar1[-1].cpu().numpy())
                np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_camera_1",
                                    data=test_camera1[-1].cpu().numpy())
                fakeCamera2 = camera_gen(test_lidar2.detach())

                np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_generated_camera_2",
                                    data=fakeCamera2[-1].cpu().numpy())
                np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_lidar_2",
                                    data=test_lidar2[-1].cpu().numpy())
                np.savez_compressed(config.TRAIN.EXAMPLE_SAVE_PATH + str(epoch) + "_camera_2",
                                    data=test_camera2[-1].cpu().numpy())

        if epoch != 0 and epoch% config.TRAIN.SAVE_AT == 0 :
            print("Saving Model at ", epoch)
            save_vanilla_model(config, camera_gen, camera_disc,  optimizer_camera_gen, optimizer_camera_disc, epoch)




def main(opts):
    load_config(opts.config)
    dataloader = lidar_camera_dataloader(config)
    device = torch.device("cuda:0" if (torch.cuda.is_available() and config.NUM_GPUS > 0) else "cpu")
    train(dataloader,config, device)


if __name__ == "__main__":
    options, args = parser.parse_args()
    main(options)

