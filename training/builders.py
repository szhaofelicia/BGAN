import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from sgan.models import TrajectoryGenerator, TrajectoryDiscriminator
from sgan.models_teampos import TrajectoryGenerator as TeamPosTrajectoryGenerator, TrajectoryDiscriminator as TeamPosTrajectoryDiscriminator

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if args.use_gpu == 1:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype


def build_sgan_models(args):
    long_dtype, float_dtype = get_dtypes(args)
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        interaction_activation=args.interaction_activation,
        batch_norm=args.batch_norm)
    generator.apply(init_weights)
    discriminator = TrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type,
        interaction_activation=args.interaction_activation,
        activation=args.d_activation  # default: relu
    )

    discriminator.apply(init_weights)
    return generator, discriminator


def build_team_pos_models(args, schema):
    generator = TeamPosTrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        tp_dropout=args.tp_dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm,
        team_embedding_dim=args.team_embedding_dim,
        pos_embedding_dim=args.pos_embedding_dim,
        interaction_activation=args.interaction_activation,
        pos_vec_len=len(schema['positions']),
        team_vec_len=3 if schema['with_ball'] else 2
    )

    generator.apply(init_weights)

    discriminator = TeamPosTrajectoryDiscriminator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        h_dim=args.encoder_h_dim_d,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tp_dropout=args.tp_dropout,
        batch_norm=args.batch_norm,
        d_type=args.d_type,
        activation=args.d_activation,  # default: relu,
        pos_embedding_dim=args.pos_embedding_dim,
        team_embedding_dim=args.team_embedding_dim,
        interaction_activation=args.interaction_activation,
        pos_vec_len=len(schema['positions']),
        team_vec_len=3 if schema['with_ball'] else 2
    )

    discriminator.apply(init_weights)
    return generator, discriminator


def build_models(args, schema, model_class="sgan"):
    if model_class == "sgan":
        return build_sgan_models(args)
    elif model_class == "team_pos":
        return build_team_pos_models(args, schema)


def build_optimizers(args, generator, discriminator):
    optimizer_g = optim.Adam(generator.parameters(), lr=args.g_learning_rate)
    optimizer_d = optim.Adam(
        discriminator.parameters(), lr=args.d_learning_rate
    )
    return optimizer_g, optimizer_d


def build_schedulers(args, optimizer_g, optimizer_d):
    scheduler_g = optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[10, 50], gamma=args.g_gamma)
    scheduler_d = optim.lr_scheduler.MultiStepLR(optimizer_d, milestones=[10, 50], gamma=args.d_gamma)
    return scheduler_g, scheduler_d


