import random

import torch
from torch import optim

import params
from generator import Generator
from layers import Discriminator
from ops_data import export_losses, export_tensors
from ops_util import bbox2mask, brush_stroke_mask, random_bbox, normalize_tensor
from ops_visual import plot_losses, plot_masks, plot_images, plot_gan_loss


class GAN:
    def __init__(self):
        self.device = params.device
        self.dtype = params.dtype
        self.gen = Generator().to(self.device)
        self.dis = Discriminator().to(self.device)

        # Hyperparameters
        self.num_epochs = params.num_epochs
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.beta1 = params.beta1
        self.beta2 = params.beta2
        self.l1_loss_alpha = params.l1_loss_alpha

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.dis.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizerG = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def train_gan(self, dataloader):
        G_losses = []
        D_losses = []
        D_losses_real = []
        D_losses_fake = []
        L_losses = []
        ex_masks = []
        ex_images = []
        iters = 0
        self.gen.train()
        self.dis.train()

        for epoch in range(self.num_epochs):
            for i, batch_data in enumerate(dataloader, 0):
                # Prepare batch
                batch_real = normalize_tensor(batch_data[0][:, 0:3, :, :]).to(self.device)
                bbox = random_bbox()
                regular_mask = bbox2mask(bbox)
                irregular_mask = brush_stroke_mask()
                mask = random.choice([regular_mask, irregular_mask]).to(self.device)
                batch_incomplete = (batch_real * (torch.tensor(1., device=self.device) - mask)).to(self.device)
                xin = batch_incomplete

                # Discriminator forward pass and GAN loss
                self.dis.zero_grad()
                # Generator output
                x1, x2, _ = self.gen(xin, mask)
                batch_predicted = x2
                batch_fake = (batch_predicted * mask + batch_incomplete * (torch.tensor(1.) - mask)).to(self.device)
                batch_real_with_masks = torch.cat((batch_real, torch.cat((mask,) * batch_real.shape[0])), dim=1)
                batch_fake_with_masks = torch.cat((batch_fake, torch.cat((mask,) * batch_fake.shape[0])), dim=1)
                # Discriminator output for both real and fake images
                labels_pos = self.dis(batch_real_with_masks)
                labels_neg = self.dis(batch_fake_with_masks.detach())
                hinge_pos = torch.mean(torch.nn.functional.relu(1 - labels_pos)).to(params.device)
                hinge_neg = torch.mean(torch.nn.functional.relu(1 + labels_neg)).to(params.device)
                D_losses_real.append(hinge_pos.item())
                D_losses_fake.append(hinge_neg.item())
                d_loss = (torch.tensor(.5, device=params.device) * hinge_pos +
                          torch.tensor(.5, device=params.device) * hinge_neg)
                dis_loss = d_loss
                D_losses.append(d_loss.item())
                d_loss.backward()
                self.optimizerD.step()

                # Generator l1 and GAN loss
                self.gen.zero_grad()
                l1_loss = self.l1_loss_alpha * (torch.mean(torch.abs(batch_real - x1)) +
                                                torch.mean(torch.abs(batch_real - x2)))
                L_losses.append(l1_loss.item())
                # Discriminator output for only fake images
                labels_neg = self.dis(batch_fake_with_masks)
                g_loss = -torch.mean(labels_neg)
                gen_loss = g_loss
                g_loss = g_loss.to(self.device)
                G_losses.append(g_loss.item())
                g_loss = g_loss + l1_loss
                g_loss.backward()
                self.optimizerG.step()

                if iters % params.iter_print == 0:
                    print("Epoch {:2d}/{:2d}, iteration {:<4d}: g_loss = {:.4f}, d_loss = {:.4f}, "
                          "d_loss(real) = {:.4f}, d_loss(fake) = {:.4f}, l1_loss = {:.4f}"
                          .format(epoch + 1, self.num_epochs, iters, gen_loss.item(), dis_loss.item(),
                                  hinge_pos.item(), hinge_neg.item(), l1_loss.item()))
                if iters % params.iter_save == 0:
                    ex_masks.append(mask)
                    ex_images.append(x2[0])
                iters += 1

        plot_losses(G_losses, D_losses, L_losses)
        plot_gan_loss(G_losses, D_losses_real, D_losses_fake)
        plot_masks(ex_masks)
        plot_images(ex_images)

        export_losses(G_losses, D_losses, D_losses_real, D_losses_fake, L_losses)
        export_tensors(ex_masks, params.ex_masks_path)
        export_tensors(ex_images, params.ex_images_path)
