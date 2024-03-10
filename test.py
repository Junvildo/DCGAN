import torch
from model import *

input = torch.rand(1,100)
ch_in=1
gen = Generator(ch_in)
weights_init(gen)
output = gen(input)
print(output.shape)

dis = Discriminator(ch_in)
weights_init(dis)
pred = dis(output)
print(pred)
print(torch.argmax(torch.softmax(pred, dim=1), dim=1))

import numpy as np
from PIL import Image
torch.manual_seed(0)
if ch_in == 1:
    image_array = output.squeeze(0).squeeze(0).detach().numpy()
else:
    image_array = output.squeeze(0).permute(1, 2, 0).detach().numpy()

image = Image.fromarray((image_array * 255).astype(np.uint8))

# Show the image
image.show()


