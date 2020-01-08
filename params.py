# cpu/gpu
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data
data_root = "./places2/val/"
dtype = torch.float32
image_size = 256

# model
gen_model_path = "./gen_model.pth"
dis_model_path = "./dis_model.pth"
ch_gen = 48
ch_dis = 64
pretrained = True

# training
num_epochs = 1
batch_size = 16

lr = 2e-4
beta1 = 0.5
beta2 = 0.999
l1_loss_alpha = 1
