import timm
import torch 


model = timm.create_model('convnext_small.fb_in22k_ft_in1k_384', pretrained=True)
model_path = "convnext_small.pth"
torch.save(model.state_dict(), model_path)
