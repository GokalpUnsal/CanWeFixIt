import random

import torch
from torch import optim

from utils import bbox2mask, brush_stroke_mask, random_bbox, gan_hinge_loss
from discriminator import Discriminator
from generator import Generator


class GAN:
    def __init__(self, device):
        self.device = device
        self.dtype = torch.float32
        self.gen = Generator().to(device)
        self.dis = Discriminator().to(device)

        # Hyperparameters
        self.num_epochs = 5
        self.batch_size = 8

        self.lr = 1e-4  # 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.l1_loss_alpha = 1

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = -1

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.dis.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizerG = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

    def train(self, dataset):
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        G_losses = []
        D_losses = []
        iters = 0
        self.gen.train()
        self.dis.train()
        for epoch in range(self.num_epochs):
            for i, batch_data in enumerate(dataloader, 0):
                # Prepare batch
                batch_real = batch_data[0]
                bbox = random_bbox()
                regular_mask = bbox2mask(bbox).permute(0, 3, 1, 2)
                irregular_mask = brush_stroke_mask().permute(0, 3, 1, 2)
                mask = random.choice([regular_mask, irregular_mask])
                batch_incomplete = batch_real * (torch.tensor(1.) - mask)
                xin = batch_incomplete
                losses = {}

                # Discriminator forward pass and GAN loss
                self.dis.zero_grad()
                # Generator output
                x1, x2, _ = self.gen(xin, mask)
                batch_predicted = x2
                batch_fake = batch_predicted * mask + batch_incomplete * (torch.tensor(1.) - mask)
                batch_mixed = torch.cat([batch_real, batch_fake], dim=0)
                batch_mixed = torch.cat((batch_mixed, torch.cat((mask,) * self.batch_size * 2)), dim=1)
                # Discriminator output for both real and fake images
                labels_mixed = self.dis(batch_mixed)
                labels_pos, labels_neg = torch.split(labels_mixed, labels_mixed.shape[0] // 2)
                _, d_loss = gan_hinge_loss(labels_pos, labels_neg)
                losses['d_loss'] = d_loss.to(self.device)
                D_losses.append(losses['d_loss'])
                losses['d_loss'].backward()
                self.optimizerD.step()

                # Generator l1 and GAN loss
                self.gen.zero_grad()
                losses['ae_loss'] = self.l1_loss_alpha * (torch.mean(torch.abs(batch_real - x1)) +
                                                          torch.mean(torch.abs(batch_real - x2)))
                # Discriminator output for only fake images
                labels_neg = self.dis(batch_fake)
                g_loss = -torch.mean(labels_neg)
                losses['g_loss'] = g_loss.to(self.device)
                G_losses.append(losses['g_loss'])
                losses['g_loss'].backward()
                self.optimizerG.step()
