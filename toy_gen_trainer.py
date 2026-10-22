"""Train a conditional flow on a 2-D toy task (utils/toy_tasks.py) with one noise-data pairing.

The toys check the continuous-condition pairings (`--pairing`, readme section 6) against
C²OT's published ordering before they are used on poses. Hyperparameters follow C²OT's toy
setup: 20k iterations, batch 256, OT batch 1024 (4 network batches), Adam at 3e-4, r_tar 0.01.
The network is this repo's conditional transformer with one token, not C²OT's MLP.

    python toy_gen_trainer.py --toy moons --pairing c2ot --seed 1 --save_path checkpoints/toy_moons/seed_1/
"""
import argparse
import os
import random

import numpy as np
import torch
from omegaconf import OmegaConf

from models.conditional_flow_matching_transformer import ConditionalFlowMatchingTransformerModel
from utils.logging_utils import _Tee, git_commit
from utils.toy_tasks import DIM, TOYS, toy_batch_sampler
from utils.train_utils import PAIRINGS, Pairer, train_one_paired_minibatch

# Checkpoint-name suffix of each pairing, as in the pose trainer. 'ot' is left out: the toys
# have no discrete conditions, so it would be the same as 'global'.
TOY_PAIRINGS = ('independent', 'global', 'c2ot', 'c2ot_fixed', 'cluster')
PAIRING_SUFFIX = {'independent': '_NOOT', 'global': '_GOT', 'c2ot': '_C2OT',
                  'c2ot_fixed': '_C2OTFIX', 'cluster': '_CLUSTER'}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--toy', choices=TOYS, required=True)
    parser.add_argument('--pairing', choices=TOY_PAIRINGS, default='c2ot')
    parser.add_argument('--r_tar', type=float, default=0.01)
    parser.add_argument('--ot_batch_mult', type=int, default=4)
    parser.add_argument('--num_clusters', type=int, default=None)
    parser.add_argument('--cond_scale', type=float, default=10.0)
    parser.add_argument('--iterations', type=int, default=20_000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='checkpoints/toy/')
    return parser.parse_args()


def run_name(toy, pairing):
    """Checkpoint stem, e.g. 'toy_moons_C2OT'."""
    return f"toy_{toy}{PAIRING_SUFFIX[pairing]}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    assert set(TOY_PAIRINGS) <= set(PAIRINGS)
    os.makedirs(args.save_path, exist_ok=True)
    stem = os.path.join(args.save_path, run_name(args.toy, args.pairing))
    tee = _Tee(stem + '_log.txt')
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available()
                          else 'mps' if torch.backends.mps.is_available() else 'cpu')

    dim = DIM[args.toy]
    model_config = {'input_dim': dim, 'output_dim': dim, 'obs_dim': 1, 'hidden_dim': 128,
                    'num_layers': 4, 'num_heads': 4, 'mlp_ratio': 4.0, 'dropout': 0.0,
                    'phase_dim': 128, 'max_seq_len': 1, 'manifold': 'euclidean', 'num_obs_tokens': 1}
    training_config = {'toy': args.toy, 'pairing': args.pairing, 'r_tar': args.r_tar,
                       'ot_batch_mult': args.ot_batch_mult, 'num_clusters': args.num_clusters,
                       'cond_scale': args.cond_scale,
                       'iterations': args.iterations, 'batch_size': args.batch_size,
                       'lr': args.lr, 'seed': args.seed, 'git_commit': git_commit()}
    OmegaConf.save(OmegaConf.create({'training': training_config, 'model': model_config}),
                   stem + '_training_config.yaml')
    print(f"Toy {args.toy}, pairing {args.pairing}, seed {args.seed}, device {device}")

    model = ConditionalFlowMatchingTransformerModel(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pairer = Pairer(args.pairing, toy_batch_sampler(args.toy, device), args.batch_size,
                    ot_batch_mult=args.ot_batch_mult, manifold='euclidean', ot_mode='flat',
                    r_tar=args.r_tar, num_clusters=args.num_clusters, cond_scale=args.cond_scale)

    running = 0.0
    for it in range(1, args.iterations + 1):
        running += train_one_paired_minibatch(model, optimizer, pairer, n_steps=1, use_cfg=False,
                                              device=device, manifold='euclidean')
        if it % 500 == 0:
            print(f"  Iteration {it}/{args.iterations}, Loss: {running / 500:.5f}{pairer.describe()}")
            running = 0.0

    model.save_checkpoint(stem + '_final.pt', optimizer=optimizer, epoch=args.iterations)
    print(f"Saved {stem}_final.pt")
    tee.close()


if __name__ == '__main__':
    main()
