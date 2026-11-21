# FPTM leakage detection

Code and raw results for the paper "Interpretable Side Channel Leakage Assessment with the Fuzzy-Pattern Tsetlin Machine" by Itamar Levi, Artem Hnilov, Alexander Fish and Leonid Yavits (manuscript in preparation).

A Fuzzy-Pattern Tsetlin Machine (FPTM) classifier is trained on threshold-coded side channel traces. Leakage is detected with a one-sided binomial test on held-out accuracy (p < 1e-5). The included clause literals rank the leaking trace samples. The repository compares the detector with Welch and chi-squared tests, MV-dCov, DL-LA (with sensitivity analysis), second-order Welch and exhaustive bivariate Welch on four public trace sets.

## Contents

| Folder | Content |
|---|---|
| `code/` | Python, Julia and shell scripts: dataset building, FPTM detection, baselines, sweeps, localization, clause analysis, ablation, timing, figures, reference checks |
| `results/` | Raw outputs behind every number in the paper: per-subset sweep results (`*_sweep.jsonl`, `*_fptm.jsonl`), localization (`loc_*.json`, `sa_*.json`, `ascad_classic_loc.json`, `locn_ascad.json`), clause dumps and statistics (`ascad_clause*`), timing (`timing_*`), GainP case study (`gainp_*`, `aggregate_gainp.json`, noiseless ground truth `gainp_gt_noiseless.npz`), trained models used for the GainP inference model (`models/`), run logs |

Trace data are not included. The scripts download them from the original sources.

## Requirements

* Python 3 with NumPy 2.5, SciPy 1.18, PyTorch 2.15 (CPU), dcor 0.7, numba 0.68, h5py 3.16, matplotlib 3.11, remotezip 0.12.
* Julia 1.13.4.
* Tsetlin.jl by Artem Hnilov, commit `0ad6747`:
  ```
  git clone https://github.com/BooBSD/Tsetlin.jl src/Tsetlin.jl
  git -C src/Tsetlin.jl checkout 0ad6747
  ```
* Optional, only for `code/check_baselines.py`: the Chowdhury and Oswald MV-dCov artifact, cloned to `src/mvld` from https://github.com/Palash123-4/Multivariate-Non-profiled-Leakage-Detection.

The full campaign took about six hours on two CPU cores and needed about 20 GB of disk at peak.

## Paths

All scripts resolve paths against one root folder:

| Variable | Default | Meaning |
|---|---|---|
| `FPTM_ROOT` | repository root | Holds `code/`, `results/`, `dl/` (downloads), `data/` (pooled sets), `work/` (intermediate files), `src/` (Tsetlin.jl) |
| `TSETLIN_JL` | `$FPTM_ROOT/src/Tsetlin.jl` | Tsetlin.jl checkout |
| `JULIA` | `julia` | Julia executable |

`dl/`, `data/`, `work/` and `src/` are ignored by git. Run Julia with `-O3 -t 2`.

## Data

* **Ledger ML-KEM EM traces** (A. A. Magazin, K. M. Abdellatif, IACR ePrint 2026/1851, LGPL v3.0). One 12.6 GB zip: https://pub-3483b4c265914de3a2a27c0a3b2076ee.r2.dev/d0nj0n_mlkem_dataset.zip. The host accepts range requests, so single members can be extracted without the full download:
  * `python3 code/list_zip.py URL` lists the members.
  * `python3 code/get_files.py URL dl/ledger MEMBER...` extracts the variable key chunks 0 and 1 of `pqm4-ref/pqm4` and `pqm4-ref/ref`.
  * `python3 code/get_tar.py URL MEMBER dl/mkm4/var.tar.gz` copies the masked (mkm4) variable key tarball.
* **ASCAD v1** (Benadjila et al., J. Cryptogr. Eng. 2020): https://static.data.gouv.fr/resources/ascad/20180530-163000/ASCAD_data.zip. Extract `ASCAD_data/ASCAD_databases/ASCAD.h5` to `dl/ascad/` with `code/get_files.py`.

## Pipeline

1. **Pooled sets.** `python3 -I code/build_datasets.py pqm4 ref mkm4 ascad` aligns the traces, computes the partition labels and the SNR ground truth, and writes `data/<set>/`.
2. **Localization inputs.** `python3 -I code/prep_fptm.py <set> 3` writes the balanced split, the thermometer-coded inputs (K = 3) and the held-out Welch t to `work/<set>_K3`.
3. **Localization runs.** `code/run_loc.sh` trains FPTM with seeds 1 to 3 and the permuted-label control (`code/fptm_detect.jl`). Then `python3 -I code/loc_eval.py <set> work/<set>_K3 3 1,2,3`. `loc_eval.py` writes `results/loc_<set>.json`, so run it on `work/<set>_K3` after any ablation run.
4. **Traces-to-detection sweeps.** `code/run_sweeps.sh` runs the classical tests, MV-dCov and DL-LA (`code/sweep.py`). `code/run_fptm_sweeps.sh` and `code/run_fptm_rest.sh` run FPTM on the same subsets (`code/fptm_sweep2.jl`). `python3 -I code/aggregate.py` writes `results/aggregate.json`.
5. **Extended mkm4 sweep** (100,000 traces). `code/run_mkm4big.sh`. `code/mkm4_poi_biv.py` runs the bivariate test on the highest-SNR share samples.
6. **Localization comparisons.** `code/dlla_sa.py <set> <epochs>` (DL-LA sensitivity analysis), `code/ascad_classic_loc.py` (bivariate Welch and chi-squared localization), `code/run_round4.sh` (localization versus trace count and the threshold ablation K = 1, 5).
7. **Clause analysis.** `julia -t 2 code/dump_clauses.jl WORKDIR SEED EPOCHS OUTFILE`, then `python3 -I code/clause_null.py` (permutation test on the share cores).
8. **Cost.** `code/timing.py <set> <epochs>` and `code/timing_fptm.jl WORKDIR`.
9. **Figures.** `python3 -I code/figs_paper.py OUTDIR ttd,loc_unmasked,loc_ascad`.
10. **GainP case study** (simulated, Section 4 of the paper).
    * `code/gainp_sim.py`: functional model of the GainP associative processor (compare, write, per-cycle energy).
    * `code/gainp_mlkem.py`: ML-KEM modular product kernel on the model. Running it checks the result against `a*b mod q`.
    * `python3 -I code/gainp_traces.py build 50000`, then `python3 -I code/gainp_traces.py make SIGMA` for SIGMA = 250, 500, 1000, 2000 (fJ): fixed key versus random key traces in `data/gainp_s<SIGMA>/`.
    * `code/run_gainp.sh`: traces-to-detection sweeps. `python3 -I code/agg_gainp.py` writes `results/aggregate_gainp.json`.
    * `python3 -I code/prep_fptm.py gainp_s1000 3`, then `code/run_gainp_loc.sh gainp_s1000`: localization runs, DL-LA sensitivity analysis, `code/loc_eval_gainp.py` (cycle-exact ground truth) and FPTM inference on GainP.
    * `code/dump_model.jl` and `code/gainp_fptm.py`: FPTM inference on the GainP model for any trained model, with a check that votes and predictions equal the Julia implementation (`results/gainp_fptm_<set>.json`).
    * `python3 -I code/fig_gainp.py OUTDIR`: GainP figure.
11. **References.** `code/verify_refs.py` pulls Crossref and arXiv metadata for every reference (`results/reference_metadata_raw.json`).

Scripts from the first feasibility run (`classic.py`, `drift.py`, `shift.py`, `align.py`, `prep.py`, `fptm_sweep.jl`, `sweep_idx.py`, `sweep_t.py`, `figs.py`, `loc.py`, `inspect_ledger.py`, `inspect2.py`, `smoke.jl`, `check_*.py`, `check_pred.jl`) are kept for completeness. The paper uses the pipeline above.

## Configuration used in the paper

FPTM: 200 clauses per polarity, T = 100, s = 100, L = 16, LF = 8, 256 states, include limit 128. Thermometer code with K = 3 per-sample quantile thresholds from the training traces. Decision: one-sided exact binomial test on held-out accuracy at p < 1e-5.
