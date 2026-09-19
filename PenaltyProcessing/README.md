# Penalty Processing Pipeline

This pipeline matches 20 Hz League trajectories to
300 Hz MoCap penalty throws.
It detects the League point of release (PoR), converts both sources to a common
coordinate system, extracts comparable features, ranks League candidates, and
reconstructs a selected continuation at 300 Hz.

Run every command below from `PenaltyProcessing/`.

## Setup

Create or activate a Python environment, then install the project and test dependencies:

```bash
python -m pip install -r requirements.txt
```

## Required data

The raw tracking data is not committed. Supply it with this layout:

```text
PenaltyProcessing/
├── penalties.csv
├── games_position_files/
│   └── <home>_vs_<away>_2_phases_positions.csv
└── mocap_files/
    └── <throw_type>/
        ├── 6DOF/       # ball marker TSV
        ├── skeleton/   # skeleton TSV
        └── body/       # optional body marker TSV
```

`penalties.csv` is semicolon-separated. The League position files must contain
ball coordinates and timestamps; the loaders report missing required columns.
Generated files are written below `out/`, which is gitignored.

## Reproduce the pipeline

### 1. Detect League release points

```bash
python src/release_detector_trajectory_based.py \
  --penalties penalties.csv \
  --positions-dir games_position_files \
  --exclude-deflections
```

The command creates a timestamped directory such as
`out/run_20260816_144549/`. Pass that run's
`simple_penalty_trajectories.csv` to the next command.
`simple_penalty_errors.csv` contains rows that could not be processed.

### 2. Build unified trajectories and features

```bash
python src/create_throw_representation.py \
  --throw-type throw_ul throw_gegendreher throw_heber throw_or \
  --league-csv out/run_YYYYMMDD_HHMMSS/simple_penalty_trajectories.csv \
  --output-dir out/throw_features
```

This writes:

```text
out/throw_features/raw_mocap.csv
out/throw_features/raw_league.csv
out/throw_features/throw_index.csv
out/throw_features/features_mocap.csv
out/throw_features/features_league.csv
out/throw_features/direction_recomputation_check.csv
```

The feature files are produced by default. League and Mocap kinematics are
recomputed from positions with the same finite-difference method.

### 3. Run the manually weighted kNN baseline

```bash
python -m src.model.weighted_knn \
  --mocap out/throw_features/features_mocap.csv \
  --league out/throw_features/features_league.csv \
  --weight-preset best_random \
  --output out/throw_features/weighted_knn_matches.csv \
  -k 10
```

Use `python -m src.model.weighted_knn --help` to inspect the other presets or
pass a JSON weight dictionary with `--weights`.

### 4. Reproduce the learned-ranker experiment

Build the deterministic synthetic train, validation, and test splits:

```bash
python -m src.ranking.build_ranking_dataset \
  --league out/throw_features/features_league.csv \
  --raw-league out/throw_features/raw_league.csv \
  --output-dir out/throw_features/ranking_dataset \
  --augmentations 6 \
  --severity-mix 0.35,0.40,0.25 \
  --hard-negatives 12 \
  --random-negatives 3 \
  --weight-preset best_random \
  --seed 42
```

Train and evaluate the pairwise ranker:

```bash
python -m src.ranking.evaluate_ranker \
  --league out/throw_features/features_league.csv \
  --mocap out/throw_features/features_mocap.csv \
  --dataset-dir out/throw_features/ranking_dataset \
  --output-dir out/throw_features/learned_ranker_all_types \
  --regularization-grid 0,0.01,0.1,1,10 \
  --minimum-weight 0.01 \
  --minimum-common-trajectory-points 2 \
  --minimum-overlap-ratio 0.5 \
  --weight-preset best_random \
  -k 10
```

`metrics.json` contains Recall@1/3/5 and MRR for the synthetic splits.
`learned_ranked_candidates.csv` and `manual_ranked_candidates.csv` contain the
real Mocap-to-League rankings.

For the recorded experiment, the learned test metrics were Recall@1 `0.5725`,
Recall@3 `0.6623`, Recall@5 `0.7087`, and MRR `0.6387`. The corresponding
manual-kNN values were `0.5145`, `0.6362`, `0.6942`, and `0.5986`.

### 5. Reproduce the manual nDCG evaluation

Create blinded relevance judgments from the union of both top-ten lists:

```bash
python -m src.ranking.manual_annotation_tool \
  --mocap out/throw_features/raw_mocap.csv \
  --league out/throw_features/raw_league.csv \
  --knn-results out/throw_features/learned_ranker_all_types/manual_ranked_candidates.csv \
  --ranker-results out/throw_features/learned_ranker_all_types/learned_ranked_candidates.csv \
  --output out/throw_features/manual_relevance_annotations.csv \
  --top-k 10 \
  --seed 42
```

The tool saves after every rating and can be resumed. Then rerun the command
from step 4 with:

```text
--manual-relevance out/throw_features/manual_relevance_annotations.csv
--exclude-mocap-throws throw_ul_seg3
```

`throw_ul_seg3` was excluded because it is a false throw detection. The
original judgments are not distributed; recreating them requires repeating
the blinded annotation. The recorded nDCG@10 values were `0.5490` for the
learned ranker and `0.4843` for manual weighted kNN.

### 6. Reconstruct the selected continuations

```bash
python src/trajectory_reconstruction.py \
  --matches out/throw_features/learned_ranker_all_types/learned_ranked_candidates.csv \
  --raw-league out/throw_features/raw_league.csv \
  --raw-mocap out/throw_features/raw_mocap.csv \
  --output out/throw_features/learned_ranker_all_types/reconstructed_rank1.csv \
  --target-hz 300 \
  --rank 1 \
  --skip-invalid
```

Prepend the measured pre-release Mocap samples:

```bash
python src/combine_full_reconstructed_trajectories.py \
  --reconstructed out/throw_features/learned_ranker_all_types/reconstructed_rank1.csv \
  --raw-mocap out/throw_features/raw_mocap.csv \
  --output out/throw_features/learned_ranker_all_types/reconstructed_rank1_full.csv
```

If the original Mocap 6DOF files are available, replace fitted pre-release ball
centres with their ground-truth centres (take note that they may not reflect the geometric center of the rigid body):

```bash
python src/combine_gt_centers_with_reconstructed_trajectories.py \
  --reconstructed-full out/throw_features/learned_ranker_all_types/reconstructed_rank1_full.csv \
  --throw-index out/throw_features/throw_index.csv \
  --mocap-root mocap_files \
  --output out/throw_features/learned_ranker_all_types/reconstructed_rank1_full_gt_ball.csv
```

## Inspect results

Plot one reconstructed match by using IDs present in the reconstruction CSV:

```bash
python visualization/plot_reconstructed_match.py \
  --reconstructed out/throw_features/learned_ranker_all_types/reconstructed_rank1.csv \
  --raw-league out/throw_features/raw_league.csv \
  --raw-mocap out/throw_features/raw_mocap.csv \
  --mocap-throw-id throw_ul_seg1 \
  --league-throw-id 12039519 \
  --output out/reconstructed_match.png
```

Every executable exposes its current options through `--help`.

## Tests

From this directory, run:

```bash
python -m pytest tests
```
