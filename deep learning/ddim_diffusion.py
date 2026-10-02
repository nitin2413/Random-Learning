# https://chatgpt.com/c/6977597a-fd38-8320-acc4-eceb3610aab7
# why do we need DDIM ?
# as DDPM in reversed process add random noise , it always create new image from same seed noise
# whereas DDIM remove that ramdom addition of noise and have different formula for reverse process
# also DDIM is faster , as it skip T (time_step) kind of momentum(if we know the direction accelerate lr with time) in optimizer.
# we use ddim when we want fast result , in this we don't add noise at reversed process because we have already find the direction of the image therefor heavy noise reduction happen which make it faster (as momentum optimizer when we know direction we increase momentum)
# same seed value with different parameter will give different output (Same seed + slightly different prompt)

def ddim_time_step(T , ddim_steps):
    return torch.linspace(0 , T-1 , ddim_steps).long() # skip time_step (1000 , 900 , 800 ..) (the momentum part).

@ torch.no_grad()
# eta is deterministic factor 0.0 no random noise at all 0.1 some amount of random ness
def genrate_image_ddip(model , batch_size= 8 ,img_size = 64 , ddim_step= 50 , eta = 0.0):
    model.eval()
    x = torch.randn(batch_size , 3, img_size , img_size , device=device)
    time_step = ddim_time_step(T , ddim_step)
    # ddim start from 1 as we calculate t_prev and for 0 will not have t_prev
    # need for t_prev as we skip time_step (900,800) t_prev might be far away ,We must explicitly know where to land
    for i in reversed(range(1 , len(time_step))):
        t = time_step[i]
        t_prev = time_step[i-1]
        # It tells the model “all images in this batch are currently at the same noise level t.”
        # at time t their can be multiple image
        t_batch = torch.full((batch_size,), t , device=device)
        noise_pred = model(x , t_batch)

        alpha_bar_t = alpha_bars[t].to(device=device)
        alpha_bar_t_prev = alpha_bars[t_prev].to(device=device)
# Recover predicted clean image x0 by inverting the forward (math formula) diffusion equation
        x0_pred = (x - torch.sqrt(alpha_bar_t) * noise_pred) / (torch.sqrt(alpha_bar_t))
# This rebuilds the image at the previous timestep by mixing the predicted clean image with the right amount of noise.
        x = (
            torch.sqrt(alpha_bar_t_prev) * x0_pred + torch.sqrt(alpha_bar_t_prev) * noise_pred
        )

    return x.clamp(-1 , 1)

samples = genrate_image_ddip(model , batch_size=4 , ddim_step=50)