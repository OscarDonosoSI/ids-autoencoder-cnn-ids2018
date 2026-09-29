# Autoencoder versus CNN-1D for Network Traffic Intrusion Detection

Reproducible code, artefacts and results for the study

> **Autoencoder versus CNN-1D for Network Traffic Intrusion Detection: A Reproducible Comparative Evaluation with Per-Class Analysis on CSE-CIC-IDS2018**
> O. E. Donoso, M. A. Pacheco Barreto, J. P. Vargas Infante, P. Recalde, R. M. Toasa
> Maestría en Seguridad Informática, Universidad Tecnológica Israel, Quito, Ecuador
> *Journal of Cybersecurity and Privacy* — manuscript `jcp-4572356`

---

## What this repository contains

An autoencoder and a one-dimensional convolutional network are compared on CSE-CIC-IDS2018 under identical preprocessing, identical stratified partitions and a fixed seed. The study then varies, in isolation, the two configuration decisions the pipeline forces on the practitioner — the anomaly threshold of the autoencoder and the class-weighting scheme of the multiclass CNN-1D — and measures how much of the reported performance each one accounts for.

The headline result is that those two decisions move the reported score by roughly an order of magnitude more than the choice of architecture does.

| Effect | Magnitude (F1) |
|---|---|
| Autoencoder threshold, p95 → p88.5 (AUC-ROC unchanged at 0.8776) | 0.4330 → 0.5609 |
| Class weighting removed, 3 seeds per arm (F1-macro) | 0.6055 ± 0.0126 → 0.7464 ± 0.0050 |
| Spread across three model families on the identical test partition | 0.0032 – 0.0207 |

---

## Releases

| Release | Contents |
|---|---|
| **v1.0** (commit `68308eb`, 30 July 2026) | Results of record for Tables 4, 5, 6 and A2 of the manuscript: autoencoder, binary CNN-1D and multiclass CNN-1D, single seed 42. |
| **v2.0** | Adds the three-seed class-weighting ablation, the run-to-run variability analysis, the Random Forest and XGBoost baselines, the end-to-end reproduction script and this README. |

`v1.0` is preserved unchanged so that every figure cited in the published article remains retrievable at the commit the Data Availability Statement names.

---

## Quick start

```bash
git clone https://github.com/OscarDonosoSI/ids-autoencoder-cnn-ids2018.git
cd ids-autoencoder-cnn-ids2018

python -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Full reproduction, nine stages, no manual intervention
python reproducir_pipeline_completo.py --raiz ./ids2018 --etapa todo
```

Check the plan and the requirements before committing nine hours of GPU time:

```bash
python reproducir_pipeline_completo.py --raiz ./ids2018 --etapa todo --simular
```

---

## The nine stages

Each stage writes its outputs to disk and can be run on its own. A stage whose outputs already exist is skipped unless `--forzar` is passed.

| Stage | `--etapa` | What it does | Approx. time |
|---|---|---|---|
| 1 | `descarga` | Downloads the ten CSV files from the public AWS S3 bucket (no credentials needed) | 25 min |
| 2 | `limpieza` | Cleans and unifies them; asserts 11,346,511 flows and the Table 1 distribution | 55 min |
| 3 | `escalado` | Signed log transform on 54 features, RobustScaler fitted on 500,000 training records, clip to (−10, 10), materialises the scaled memmap | 35 min |
| 4 | `particion` | Stratified 70/15/15 split, seed 42 | 2 min |
| 5 | `autoencoder` | Trains on benign traffic only, sweeps percentiles 50–99.5 to calibrate the threshold | 2 h 10 min |
| 6 | `cnn` | Trains the binary and 15-class CNN-1D | 2 h 30 min |
| 7 | `baselines` | Random Forest and XGBoost on a 500,000-flow stratified subsample, evaluated on the full test partition | 40 min |
| 8 | `ablacion` | Six runs: two loss configurations × three seeds (42, 7, 123) | 2 h 20 min |
| 9 | `reportes` | Regenerates every table (CSV + Markdown) and figure (PNG, 300 dpi) from the serialised results | 3 min |

Run a single stage:

```bash
python reproducir_pipeline_completo.py --raiz ./ids2018 --etapa ablacion
```

---

## Verification built into the pipeline

The cleaning stage aborts rather than continue if the data does not match the published dataset. It checks the raw flow count (13,087,277), the cleaned flow count (11,346,511), the feature count (65), the number of imputed NaN values (3,270,477) and every one of the fifteen per-class counts in Table 1. The partition stage checks the three subset sizes (7,947,096 / 1,697,438 / 1,701,977). Any divergence surfaces at the point where it occurs instead of propagating silently into the metrics.

---

## Reproducibility and its limits

Seed 42 is fixed in Python, NumPy, PyTorch and scikit-learn before any stochastic operation, which makes the split, the scaler sample, the weight initialisation and the batch ordering reproducible. cuDNN reduction kernels are not bit-reproducible on GPU, so two identical runs can differ in the last decimals. Pass `--determinista` to enable deterministic kernels at roughly 30% additional runtime.

That residual variation is not a footnote in this study; it is one of its results. Stage 8 measures it, and the two arms behave very differently:

| | With balanced weights | Without weights |
|---|---|---|
| F1-macro | 0.6055 ± 0.0126 | 0.7464 ± 0.0050 |
| Accuracy | 0.6182 ± 0.0585 | 0.9887 ± 0.0001 |
| Recall, Benign | 0.5661 ± 0.0674 | 0.9994 ± 0.0002 |
| Benign labelled Infilteration | 42.90% ± 6.71 | 0.02% ± 0.02 |

Three runs of identical code in the weighted arm produced benign recall of 0.50 and 0.64. Reporting a single seed for that configuration is reproducible in its headline number and not reproducible in its operational meaning.

**One honest caveat.** The model weights of the original single-seed multiclass run behind Tables 4, 6 and A2 were overwritten during a later execution of the pipeline and could not be recovered. Those tables are derived from the evaluation output serialised at the time of the original run, which is published here under `results/v1.0/`. That run is documented but not re-executable weight-for-weight. This is stated in the manuscript's Limitations, and it is why the ablation was designed to be seed-explicit and fully re-executable from raw data.

---

## Repository layout

```
.
├── reproducir_pipeline_completo.py   # end-to-end pipeline, nine stages
├── requirements.txt                  # pinned dependencies
├── notebooks/
│   ├── AutoencoderVsCNN.ipynb        # original exploratory notebook (v1.0)
│   └── ablacion_pesos_de_clase.ipynb # ablation, runnable on Colab
├── results/
│   ├── v1.0/                         # results of record for the published tables
│   │   ├── autoencoder_resultados.json
│   │   ├── cnn1d_binary_resultados.json
│   │   └── cnn1d_multiclass_resultados.json
│   ├── ablacion/                     # six runs + RESUMEN_ABLACION.json
│   └── baselines/                    # random forest and xgboost
├── artifacts/                        # fitted scaler, label encoder,
│                                     # log-feature index, metadata
├── outputs/
│   ├── tables/                       # regenerated tables, CSV and Markdown
│   └── figures/                      # regenerated figures, PNG 300 dpi
└── docs/
    └── mapa_tablas.md                # which file produces which table of the article
```

Partition indices (`idx_train/val/test.npy`) are not tracked either: they total around 90 MB and are regenerated deterministically from seed 42 with `--etapa particion`. Trained weights are distributed as release assets rather than tracked in git, for the same reason of size. The `.pt` files of the ablation are attached to release `v2.0`.

---

## Hardware and runtime

| | |
|---|---|
| Platform | Google Colaboratory |
| GPU | NVIDIA Tesla T4, 15,360 MB VRAM |
| RAM | 12 GB |
| Disk | ~85 GB peak (6.5 GB raw, 1.5 GB Parquet, 2.9 GB scaled memmap, rest temporary) |
| Total runtime | ~9 h |

The 11,346,511 × 65 float32 matrix is never loaded into RAM: it is materialised as a memory-mapped array and read in contiguous slices by a lightweight loader, which is what makes the pipeline fit in 12 GB.

---

## Dataset

CSE-CIC-IDS2018 was produced by the Communications Security Establishment and the Canadian Institute for Cybersecurity on an AWS infrastructure with 50 attacking machines and 420 victim hosts, over ten capture days. It is publicly available from `s3://cse-cic-ids2018/` without authentication, and stage 1 downloads it directly. It is redistributed here in no form; only derived artefacts and results are included.

---

## Citation

```bibtex
@article{donoso2026autoencoder,
  title   = {Autoencoder versus {CNN-1D} for Network Traffic Intrusion Detection:
             A Reproducible Comparative Evaluation with Per-Class Analysis on
             {CSE-CIC-IDS2018}},
  author  = {Donoso, Oscar E. and Pacheco Barreto, Marco A. and
             Vargas Infante, Juan P. and Recalde, Pablo and Toasa, Renato M.},
  journal = {Journal of Cybersecurity and Privacy},
  year    = {2026},
  note    = {Manuscript jcp-4572356}
}
```

## License

Code released under the MIT License. The CSE-CIC-IDS2018 dataset remains subject to the terms of the Canadian Institute for Cybersecurity.
