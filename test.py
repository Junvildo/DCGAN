import torch
from model import *

input = torch.rand(1,100)

gen = Generator()
output = gen(input)
print(output.shape)

dis = Discriminator()
pred = dis(output)
print(pred)

# import numpy as np
# from PIL import Image

# # Assuming your tensor is stored in a variable called 'image_tensor'
# # Convert the tensor to a NumPy array
# image_array = output.squeeze(0).permute(1, 2, 0).detach().numpy()

# # Convert to the shape (64, 64, 3)
# # Permute is used to change the order of dimensions to (H, W, C)

# # Convert the array to an image
# image = Image.fromarray((image_array * 255).astype(np.uint8))

# # Show the image
# image.show()


