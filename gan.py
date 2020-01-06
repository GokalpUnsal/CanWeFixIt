import torch
from torch import optim
import torch.nn as nn

from utils import bbox2mask, brush_stroke_mask, random_bbox
from discriminator import Discriminator
from generator import Generator


class GAN:
    def __init__(self, device):
        self.device = device
        self.dtype = torch.float32
        self.gen = Generator().to(self.device)
        self.dis = Discriminator().to(self.device)

        # Number of training epochs
        self.num_epochs = 5

        # Learning rate for optimizers
        self.lr = 0.0002

        # Beta1 hyperparam for Adam optimizers
        self.beta1 = 0.5

        self.batch_size = 32

        # Create batch of latent vectors that we will use to visualize
        #  the progression of the generator
        # fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        # Establish convention for real and fake labels during training
        self.real_label = 1
        self.fake_label = -1

        # Setup Adam optimizers for both G and D
        self.optimizerD = optim.Adam(self.dis.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

    def train(self, dataset):
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        img_list = []
        G_losses = []
        D_losses = []
        iters = 0
        # out = self.gen(x, mask)
        # img = out[0].permute(1,2,0)
        # img = img.cpu()
        # img = img.detach().numpy()
        # img = (img /2) + 0.5
        # plt.imshow(img)
        # plt.show()
        # x = torch.randn((8, 3, 256, 256), dtype=self.dtype, device=self.device)
        # mask = torch.zeros((8, 1, 256, 256), dtype=self.dtype, device=self.device)

        for epoch in range(self.num_epochs):
            for i, batch_data in enumerate(dataloader, 0):
                # Prepare batch
                batch_pos = batch_data

                bbox = random_bbox()
                regular_mask = bbox2mask(bbox).permute(0, 3, 1, 2)
                irregular_mask = brush_stroke_mask().permute(0, 3, 1, 2)
                mask = (regular_mask.type(torch.bool) | irregular_mask.type(torch.bool)).type(torch.float32).to(self.device)
                batch_pos[0] = batch_pos[0].to(self.device)
                batch_incomplete = batch_pos[0] * (torch.tensor(1.).cuda() - mask)
                xin = batch_incomplete

                # Forward pass for generator
                self.gen.zero_grad()
                x1, x2, offset_flow = self.gen(xin, mask)
                batch_predicted = x2
                losses = {}
                # Apply mask and complete image
                batch_complete = batch_predicted * mask + batch_incomplete * (torch.tensor(1.) - mask)

                losses['ae_loss'] = torch.mean(torch.abs(batch_pos[0] - x1), dim=-1)
                losses['ae_loss'] += torch.mean(torch.abs(batch_pos[0] - x2), dim=-1)

                batch_pos_neg = torch.cat([batch_pos[0], batch_complete], dim=0)
                # TODO: torch tile
                # batch_pos_neg = torch.cat([batch_pos_neg, torch.tile(mask, [self.batch_size * 2, 1, 1, 1])], dim=3)
                # TODO: hinge loss

                # Forward pass for discriminator
                self.dis.zero_grad()
                pos_neg = self.dis(batch_pos_neg)
                pos, neg = torch.split(pos_neg, pos_neg.shape[0] // 2)
                # TODO: error sum
                # errD_real =
                # errD_real.backward()
                # D_x = output.mean().item()
                self.gen.zero_grad()


if __name__ == '__main__':
    gan = GAN()
    gan.train()
