# cpu/gpu
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data
data_root = "./imagenet/val/"
dtype = torch.float32
image_size = 64

# model
gen_model_path = "./gen_model.pth"
dis_model_path = "./dis_model.pth"
ch_gen = 32
ch_dis = 48
pretrained = False

ex_masks_path = "./ex_masks.pth"
ex_images_path = "./ex_images.pth"

# training
num_epochs = 4
batch_size = 16

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
l1_loss_alpha = 1

iter_print = 10
iter_save = 100
