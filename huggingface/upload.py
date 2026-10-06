"""Stage and upload the ablation artifacts to a single HuggingFace model repo.

Layout pushed to the hub:

  /pose/seed_<N>/*.pt + *.yaml     (6 pose checkpoints + configs per seed)
  /mnist/seed_<N>/*.pt + *.yaml    (6 MNIST checkpoints + configs per seed)
  /classifier/mnist_cnn.pt         (evaluation oracle)
  /results/<task>/epoch_<E>/*_summary.csv   (mean ± std over seeds: metrics, sweeps)
  /results/<task>/epoch_<E>/*.png           (sweep plots)
  /results/<task>/epoch_<E>/seed_<N>/       (per-seed CSVs and sample grids)
  README.md                        (rendered model card)

Authentication: run `huggingface-cli login` first, or pass --token. Use
--dry-run to print the staged tree without uploading.
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import csv

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.aggregate_seeds import METRICS


# Mirror the ckpt-name regex from experiments/evaluate_all.py.
_CKPT_RE = re.compile(
    r"^(?P<prefix>cond_)?(?P<task>pose|image)_flow_matching_model"
    r"(?P<ot>_OT|_NOOT)(?P<cfg>_CFG|_NOCFG)_epoch_(?P<epoch>\d+)\.pt$"
)


def stage_artifacts(checkpoint_dir: Path, results_dir: Path,
                    classifier_path: Path, stage_dir: Path):
    """Copy the latest-epoch checkpoint of each variant, for every seed, plus supporting
    files into stage_dir, mirroring the local checkpoints/<task>/seed_<N>/ layout."""
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / 'classifier').mkdir()

    for task in ('pose', 'mnist'):
        for seed_dir in sorted((checkpoint_dir / task).glob('seed_*')):
            # Group by ablation key, keep latest epoch.
            latest = {}
            for path in sorted(seed_dir.glob("*.pt")):
                m = _CKPT_RE.match(path.name)
                if not m:
                    continue
                key = (m['prefix'], m['task'], m['ot'], m['cfg'])
                epoch = int(m['epoch'])
                if key not in latest or epoch > latest[key][1]:
                    latest[key] = (path, epoch)

            dst_dir = stage_dir / task / seed_dir.name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for (path, _epoch) in latest.values():
                shutil.copy2(path, dst_dir / path.name)
                # Pair the .yaml config (filename has no _epoch suffix).
                stem = path.name.split("_epoch_")[0]
                config_src = path.parent / f"{stem}_training_config.yaml"
                if config_src.exists():
                    shutil.copy2(config_src, dst_dir / config_src.name)

    if classifier_path.exists():
        shutil.copy2(classifier_path, stage_dir / 'classifier' / classifier_path.name)

    if results_dir.exists():
        shutil.copytree(results_dir, stage_dir / 'results', dirs_exist_ok=True)

    return stage_dir


def _render_metrics_table(results_dir: Path):
    rows = []
    for task in ('pose', 'mnist'):
        summaries = sorted((results_dir / task).glob('epoch_*/metrics_summary.csv'),
                           key=lambda p: int(p.parent.name.split('_')[1]))
        for summary in summaries:
            with open(summary) as f:
                rows.extend(csv.DictReader(f))
    if not rows:
        return ("(no metrics_summary.csv found — run ./pose_ablations.sh eval and "
                "./mnist_ablations.sh eval first)\n")
    metrics = [m for m in METRICS if any(r.get(f'{m}_mean') for r in rows)]
    headers = ['task', 'epoch', 'variant', 'n_seeds'] + metrics
    lines = ['| ' + ' | '.join(headers) + ' |',
             '| ' + ' | '.join(['---'] * len(headers)) + ' |']
    for r in rows:
        cells = [r['task'], r['epoch'], r['variant'], r['n_seeds']]
        for m in metrics:
            mean, std = r.get(f'{m}_mean'), r.get(f'{m}_std')
            cells.append(f"{float(mean):.4f} ± {float(std):.4f}" if mean else '')
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def render_model_card(template_path: Path, results_dir: Path, output: Path,
                      repo_id: str, github_url: str):
    template = template_path.read_text()
    rendered = (template
                .replace('{{REPO_ID}}', repo_id)
                .replace('{{GITHUB_URL}}', github_url)
                .replace('{{METRICS_TABLE}}', _render_metrics_table(results_dir)))
    output.write_text(rendered)
    print(f"Rendered model card to {output}")


def push_to_hub(stage_dir: Path, repo_id: str, token: str = None,
                commit_message: str = "Upload ablation artifacts"):
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=token)
    create_repo(repo_id=repo_id, repo_type='model', exist_ok=True, token=token)
    api.upload_folder(
        folder_path=str(stage_dir),
        repo_id=repo_id,
        repo_type='model',
        commit_message=commit_message,
    )
    print(f"Uploaded to https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_id', type=str, required=True,
                        help='HuggingFace repo (e.g. joeliechty/flow-matching-transformer-ablations)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    parser.add_argument('--results_dir', type=str, default='experiments/results/')
    parser.add_argument('--classifier_path', type=str, default='eval_assets/mnist_cnn.pt')
    parser.add_argument('--stage_dir', type=str, default='huggingface/_stage/')
    parser.add_argument('--template', type=str, default='huggingface/MODEL_CARD.md')
    parser.add_argument('--github_url', type=str,
                        default='https://github.com/joeliechty/flow_matching_transformer')
    parser.add_argument('--token', type=str, default=os.environ.get('HF_TOKEN'))
    parser.add_argument('--dry-run', dest='dry_run', action='store_true')
    args = parser.parse_args()

    stage_dir = Path(args.stage_dir)
    stage_artifacts(
        Path(args.checkpoint_dir), Path(args.results_dir),
        Path(args.classifier_path), stage_dir,
    )
    render_model_card(
        Path(args.template), Path(args.results_dir),
        stage_dir / 'README.md',
        repo_id=args.repo_id, github_url=args.github_url,
    )

    print("\nStaged tree:")
    for path in sorted(stage_dir.rglob('*')):
        if path.is_file():
            print(f"  {path.relative_to(stage_dir)}  ({path.stat().st_size:,} bytes)")

    if args.dry_run:
        print("\n--dry-run set — not uploading.")
        return
    push_to_hub(stage_dir, args.repo_id, token=args.token)


if __name__ == '__main__':
    main()
