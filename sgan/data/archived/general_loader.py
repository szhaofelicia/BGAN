from torch.utils.data import DataLoader

# from sgan.data.trajectories import TrajectoryDataset, seq_collate
# from sgan.data.trajectories_basketball import TrajectoryDataset, seq_collate
from sgan.data.trajectories_general import TrajectoryDataset, seq_collate


def data_loader(args, path, schema):
    dset = TrajectoryDataset(
        path,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        skip=args.skip,
        delim=args.delim,
        metric=args.metric,
        schema=schema
    )

    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.loader_num_workers,
        collate_fn=seq_collate)
    return dset, loader
