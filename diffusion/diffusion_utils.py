import torch
import torch.nn as nn


# T = number of diffusion time steps i.e final noise distribution is q(x_T) and data distribtion is q(x_0)
# q denotes forward noise process distributions
# p denotes backward denoising process distributions

#
# Credits: Based on tutorial https://huggingface.co/blog/annotated-diffusion
#

def build_diffusion_process(args):
    if args.noise_schedule == "linear":
        betas = linear_schedule(args.beta_start, args.beta_end, args.num_timesteps)
    elif args.noise_schedule == "cosine":
        betas = cosine_schedule(args.num_timesteps)
    elif args.noise_schedule == "quadratic":
        betas = quadratic_schedule(args.beta_start, args.beta_end, args.num_timesteps)
    else:
        raise ValueError("Invalid noise schedule")
    
    diffusion_process = DiffusionProcess(betas)
    return diffusion_process


def kl_divergence(mean_p, mean_q, logvar_p, logvar_q):
    """
    Computes the KL divergences KL(p|q) between two normal distributions with diagonal covariance matrices
    mean_p: tensor of shape (..., num_latents)
    mean_q: tensor of shape (..., num_latents)
    logvar_p: tensor of shape (..., num_latents)
    logvar_q: tensor of shape (..., num_latents)
    returns tensor of shape (...)   
    """
    kl = 0.5 * (logvar_q - logvar_p + (torch.exp(logvar_p) + (mean_p - mean_q)**2) / torch.exp(logvar_q) - 1)
    return kl.sum(dim=-1)


class DiffusionProcess(nn.Module):

    def __init__(self, betas) -> None:
        super().__init__()

        self.num_timesteps = len(betas) # T in the formulars

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', get_alphas(betas))
        self.register_buffer('sqrt_alpha_inv', torch.sqrt(1.0 / self.alphas))

        self.register_buffer('alpha_bars', get_alpha_bars(betas))
        self.register_buffer('sqrt_alpha_bar', torch.sqrt(self.alpha_bars))
        self.register_buffer('sqrt_alpha_bar_neg', torch.sqrt(1 - self.alpha_bars))
        self.register_buffer('posterior_variance',
                             self.betas * (1.0 - torch.nn.functional.pad(self.alpha_bars[:-1], (1, 0), value=1.0)) / (1.0 - self.alpha_bars) # variance of q(x_tidx - 1 | x_0 x_t)
                             )


    def q_sample(self, x_0, tidx, noise=None):
        """
        Samples from q(x_t | x_0) for a given time step t
        x_0: tensor of shape (batch_size, ...)
        t:  time step indices (batch_size,)
        noise: tensor of shape (batch_size, ...)
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bar, tidx, target_x=x_0)
        sqrt_alpha_bar_neg_t = extract(self.sqrt_alpha_bar_neg, tidx, target_x=x_0)

        return sqrt_alpha_bar_t * x_0 + sqrt_alpha_bar_neg_t * noise
    

    def p_loss(self, denoising_model, x_0, tidx, condition=None, noise=None, loss_fn='l2'):
        """
        Computes the loss for the denoising process p(x_0 | x_t) for a given time step t
        x_0: tensor of shape (batch_size, ...)
        tidx:  time step indices (batch_size,)
        condition: (optional) tensor of shape (batch_size, ...) representing the condition vector
        noise: (optional) tensor of shape (batch_size, ...)
        loss_fn: loss function to use. One of 'l1', 'l2', 'huber'
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t = self.q_sample(x_0, tidx, noise)

        if condition is None:
            pred_noise = denoising_model(x_t, tidx)
        else:
            pred_noise = denoising_model(x_t, tidx, condition)

        if loss_fn == 'l1':
            loss = torch.nn.functional.l1_loss(noise, pred_noise)
        elif loss_fn == 'l2':
            loss = torch.nn.functional.mse_loss(noise, pred_noise)
        elif loss_fn == 'huber':
            loss = torch.nn.functional.smooth_l1_loss(noise, pred_noise)
        else:
            raise ValueError("Invalid loss function")
        
        return loss
    
    @torch.no_grad()
    def p_sample_step(self, denoising_model, x_t, tidx, condition=None, no_noise=False):
        """
        Samples from p(x_{t-1} | x_t) for a given time step t
        denoising_model: denoising model
        x_t: tensor of shape (batch_size, ...)
        tidx:  time step indices (batch_size,). tidx should be time step - 1
        condition: (optional) tensor of shape (batch_size, ...) representing the condition vector
        """
        beta_t = extract(self.betas, tidx, target_x=x_t)

        sqrt_alpha_bar_neg_t = extract(self.sqrt_alpha_bar_neg, tidx, target_x=x_t)
        sqrt_alpha_inv_t = extract(self.sqrt_alpha_inv, tidx, target_x=x_t)

        if condition is None:
            mean = sqrt_alpha_inv_t * (x_t - beta_t * denoising_model(x_t, tidx) / sqrt_alpha_bar_neg_t)
        else:
            mean = sqrt_alpha_inv_t * (x_t - beta_t * denoising_model(x_t, tidx, condition) / sqrt_alpha_bar_neg_t)

        if no_noise:
            return mean
        else:
            posterior_variance_t = extract(self.posterior_variance, tidx, target_x=x_t)
            noise = torch.randn_like(x_t) * torch.sqrt(posterior_variance_t)
            return mean + noise

    @torch.no_grad()
    def p_sample(self, model, shape, condition=None, return_intermediate_steps=False):
        """
        Samples from p(x_0) given a denoising model.
        model: denoising model
        shape: shape samples (bs, ...)
        condition: (optional) tensor of shape (batch_size, ...) representing the condition vector

        Returns: 
        samples
        if return_intermediate_steps is True: samples, intermediate_steps
        """
        device = next(model.parameters()).device

        num_samples = shape[0]
        # start from pure noise (for each example in the batch)
        samples = torch.randn(shape, device=device)
        
        intermediate_steps = []
        intermediate_steps.append(samples.cpu().clone())

        for i in reversed(range(0, self.num_timesteps)):
            samples = self.p_sample_step(model, 
                                         samples, 
                                         torch.full((num_samples,), i, device=device, dtype=torch.long), 
                                         condition=condition
                                         )
            if return_intermediate_steps:
                intermediate_steps.append(samples.cpu().clone())
        if return_intermediate_steps:
            return samples, intermediate_steps
        else:
            return samples


def linear_schedule(start_value, end_value, num_steps):
    return torch.linspace(start_value, end_value, num_steps)

def quadratic_schedule(start_value, end_value, num_steps):
    return torch.linspace(start_value**0.5, end_value**0.5, num_steps) ** 2

def cosine_schedule(num_steps, s=0.008):

    x = torch.linspace(0, num_steps, num_steps+1)
    alpha_bars = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi / 2)**2
    alpha_bars = alpha_bars / alpha_bars[0]
    alpha = alpha_bars[1:] / alpha_bars[:-1]
    beta = 1 - alpha
    beta = torch.clip(beta, 0.0001, 0.9999)
    
    return beta

def get_alphas(betas):
    """
    betas: tensor of variance schedule of shape (num_steps,)
    """
    alphas = 1 - betas
    return alphas

def get_alpha_bars(betas):
    alphas = get_alphas(betas)
    alpha_bars = torch.cumprod(alphas, 0)
    return alpha_bars

def extract(schedule, tidx, target_x=None):
    """
    tidx: tensor of shape (batch_size). tidx should be time steps - 1
    target_x: tensor of shape (batch_size, ...) to which we will later apply the schedule. Resulting tensor will have the same number of squeezed trailing dimensions.
    returns tensor of shape (batch_size) with values from schedule at indices t
    """
    if target_x is not None:
        return schedule[tidx].reshape(target_x.shape[0], *([1] * len(target_x.shape[1:])))
    else:
        return schedule[tidx]
    
