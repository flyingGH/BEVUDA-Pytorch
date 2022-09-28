from layers.discriminator.img_source import Disc_img_source
from layers.discriminator.img_target import Disc_img_target
import torch

aaa = Disc_img_target()

input = torch.rand([4, 6, 512, 16, 44])
label = torch.rand(4,1).long()

aaa.train_target(input, label, aaa.parameters(), 1e-5)

print('test git scc')