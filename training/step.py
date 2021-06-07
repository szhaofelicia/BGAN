import torch
import torch.nn as nn
from sgan.utils import relative_to_abs, get_dset_path
from sgan.losses import gan_g_loss, gan_d_loss, l2_loss


def discriminator_step(
    args, batch, generator, discriminator, d_loss_fn, optimizer_d
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     obs_team_vec, obs_pos_vec, pred_team_vec, pred_pos_vec,
     non_linear_ped, loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, obs_team_vec, obs_pos_vec)

    pred_traj_fake_rel = generator_out
    pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

    traj_real = torch.cat([obs_traj, pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel, pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    all_team_vec = torch.cat([obs_team_vec, pred_team_vec], dim=0)
    all_pos_vec = torch.cat([obs_pos_vec, pred_pos_vec], dim=0)
    scores_fake = discriminator(traj_fake, traj_fake_rel, all_team_vec, all_pos_vec, seq_start_end)
    scores_real = discriminator(traj_real, traj_real_rel, all_team_vec, all_pos_vec, seq_start_end)


    # Compute loss with optional gradient penalty
    # data_loss = d_loss_fn(scores_real, scores_fake)
    # losses['D_data_loss'] = data_loss.item()
    # loss += data_loss

    d_loss_real, d_loss_fak=d_loss_fn(scores_real, scores_fake)
    losses['D_real_loss'] = d_loss_real.item()
    losses['D_fake_loss'] = d_loss_fak.item()
    loss += d_loss_real+d_loss_fak

    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    if args.clipping_threshold_d > 0:
        nn.utils.clip_grad_norm_(discriminator.parameters(),
                                 args.clipping_threshold_d)
    optimizer_d.step()

    return losses

def generator_step(
    args, batch, generator, discriminator, g_loss_fn, optimizer_g
):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
     obs_team_vec, obs_pos_vec, pred_team_vec, pred_pos_vec,
     non_linear_ped, loss_mask, seq_start_end) = batch
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = []

    loss_mask = loss_mask[:, args.obs_len:]

    for _ in range(args.best_k):
        generator_out = generator(obs_traj, obs_traj_rel, seq_start_end, obs_team_vec, obs_pos_vec)


        pred_traj_fake_rel = generator_out
        pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1])

        if args.l2_loss_weight > 0:
            g_l2_loss_rel.append(args.l2_loss_weight * l2_loss(
                pred_traj_fake_rel,
                pred_traj_gt_rel,
                loss_mask,
                mode=args.l2_loss_mode # default:"raw"
            ))

    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    if args.l2_loss_weight > 0:
        g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)
        for start, end in seq_start_end.data:
            _g_l2_loss_rel = g_l2_loss_rel[start:end]
            _g_l2_loss_rel = torch.sum(_g_l2_loss_rel, dim=0)
            _g_l2_loss_rel = torch.min(_g_l2_loss_rel) / torch.sum(
                loss_mask[start:end])
            g_l2_loss_sum_rel += _g_l2_loss_rel
        losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()
        loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj, pred_traj_fake], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel, pred_traj_fake_rel], dim=0)
    all_team_vec = torch.cat([obs_team_vec, pred_team_vec], dim=0)
    all_pos_vec = torch.cat([obs_pos_vec, pred_pos_vec], dim=0)
    scores_fake = discriminator(traj_fake, traj_fake_rel, all_team_vec, all_pos_vec,seq_start_end)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    if args.clipping_threshold_g > 0:
        nn.utils.clip_grad_norm_(
            generator.parameters(), args.clipping_threshold_g
        )
    optimizer_g.step()

    return losses
