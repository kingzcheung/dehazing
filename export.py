from safetensors.torch import save_model
import torch

from net import dehaze_net


model = dehaze_net()
model.load_state_dict(torch.load('dehazer.pth',map_location='cpu'))
model.eval()
model.cpu()
print(model)
# export to safetensors
save_model(model, 'dehazer.safetensors')
