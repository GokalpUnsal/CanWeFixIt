import random

import torch
from torch import optim

import params
from generator import Generator
from layers import Discriminator
from ops_util import bbox2mask, brush_stroke_mask, random_bbox, gan_hinge_loss
from ops_visual import plot_losses, display_tensor_mask


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
        L_losses = []
        iters = 0
        self.gen.train()
        self.dis.train()

        # bbox = random_bbox()
        # mask = bbox2mask(bbox).to(self.device)
        # display_tensor_mask(mask)

        for epoch in range(self.num_epochs):
            for i, batch_data in enumerate(dataloader, 0):
                # Prepare batch
                batch_real = batch_data[0].to(self.device)
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
                batch_mixed = torch.cat([batch_real, batch_fake], dim=0).to(self.device)
                batch_mixed = torch.cat((batch_mixed, torch.cat((mask,) * batch_mixed.shape[0])), dim=1).to(self.device)
                # Discriminator output for both real and fake images
                labels_mixed = self.dis(batch_mixed)
                labels_pos, labels_neg = torch.split(labels_mixed, labels_mixed.shape[0] // 2)
                _, d_loss = gan_hinge_loss(labels_pos, labels_neg)
                dis_loss = d_loss
                D_losses.append(d_loss.item())
                d_loss.backward(retain_graph=True)
                self.optimizerD.step()
                # Generator l1 and GAN loss
                self.gen.zero_grad()
                l1_loss = self.l1_loss_alpha * (torch.mean(torch.abs(batch_real - x1)) +
                                                torch.mean(torch.abs(batch_real - x2)))
                L_losses.append(l1_loss.item())
                # Discriminator output for only fake images
                batch_fake = torch.cat((batch_fake, torch.cat((mask,) * batch_fake.shape[0])), dim=1)
                labels_neg = self.dis(batch_fake)
                g_loss = -torch.mean(labels_neg)
                gen_loss = g_loss
                g_loss = g_loss.to(self.device)
                G_losses.append(g_loss.item())
                g_loss = g_loss + l1_loss
                g_loss.backward()
                self.optimizerG.step()
                if iters % 10 == 0:
                    print("Epoch {:2d}/{:2d}, iteration {:<4d}: g_loss = {:.5f}, d_loss = {:.5f}, l1_loss = {:.5f}"
                          .format(epoch + 1, self.num_epochs, iters, gen_loss.item(), dis_loss.item(), l1_loss.item()))
                iters += 1

        plot_losses(G_losses, D_losses, L_losses)
