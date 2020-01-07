# cpu/gpu
device = None

# data
image_size = 256
data_root = "./places2/"

# model
model_path = "./model.pth"
ch_gen = 48
ch_dis = 64

# training
num_epochs = 10
batch_size = 4

lr = 1e-4
beta1 = 0.5
beta2 = 0.999
l1_loss_alpha = 1
