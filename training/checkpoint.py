import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

def initialize_checkpoint(args):
    checkpoint = {
        'args': args.__dict__,
        'G_losses': defaultdict(list),
        'D_losses': defaultdict(list),
        'losses_ts': [],
        'metrics_val': defaultdict(list),
        'metrics_train': defaultdict(list),
        'sample_ts': [],
        'restore_ts': [],
        'norm_g': [],
        'norm_d': [],
        'counters': {
            't': None,
            'epoch': None,
        },
        'g_state': None,
        'g_optim_state': None,
        'd_state': None,
        'd_optim_state': None,
        'g_best_state': None,
        'd_best_state': None,
        'best_t': None,
        'g_best_nl_state': None,
        'd_best_state_nl': None,
        'best_t_nl': None,
    }
    return checkpoint

def restore_from_checkpoint(checkpoint, generator, discriminator, optimizer_g, optimizer_d):
    generator.load_state_dict(checkpoint['g_state'])
    discriminator.load_state_dict(checkpoint['d_state'])
    optimizer_g.load_state_dict(checkpoint['g_optim_state'])
    optimizer_d.load_state_dict(checkpoint['d_optim_state'])
    t = checkpoint['counters']['t']
    epoch = checkpoint['counters']['epoch']
    return generator, discriminator, optimizer_g, optimizer_d, t, epoch
