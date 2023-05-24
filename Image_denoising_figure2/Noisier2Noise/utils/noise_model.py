import torch

def get_noise(data, noise_seed, fix_noise, noise_std = float(25)/255.0):

    
    if fix_noise:
        device = torch.device('cuda')
        gen = torch.Generator(device=device)
        batch_size = data.size(dim=0)
        tensor_dim = list(data.size())[1:]
        
        for i in range(0,batch_size):
            gen = gen.manual_seed(noise_seed[i].item())
            noise =  torch.randn(tensor_dim,generator = gen, device=device) * noise_std
            noise = torch.unsqueeze(noise,0)
            if i == 0:
                noise_tensor = noise
            else:
                noise_tensor = torch.cat((noise_tensor, noise),0)

        noise = noise_tensor
        #noise =  torch.randn(data.shape,generator = gen, device=device) * noise_std
    else:
        noise = torch.randn_like(data)
        noise.data = noise.data * noise_std

    return noise