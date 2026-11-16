# Motif: does a DNA language model know whether a motif works in its context?

Experiments shown at the 2026-10-27 group meeting, plus the notebooks that produce their inputs.

**Question.** A transcription-factor motif in DNA is sometimes bound and sometimes not. Do the hidden states
of a DNA language model at the motif's position carry that information? We fit a linear direction (the
**function CAV**) that separates ChIP-seq-bound from unbound copies of the same motif, matched on motif
strength, GC content and CpG. The direction gives each motif occurrence a probability, written P(bound).

## How to run

Every notebook runs in Colab and is self-contained: it installs its own packages and mounts Google Drive.
Data and results live on Drive, not in this repo, under

```
MyDrive/GitHub/Biological-Foundation-Model/Notebooks/EP Pair Evaluation/per_occurrence/motif_token_study/
```

## Experiments from the 2026-10-27 meeting

Run in this order; later notebooks read what earlier ones wrote to Drive.

| notebook | what it asks | status / result |
|---|---|---|
| `Partner Insertion - Does a Co-operating Motif Raise P(bound).ipynb` | Insert a known partner motif 6–150 bp from a motif. Does P(bound) rise more than when a random word of the same length is inserted? NT-v2 500M. | Run. 9 pairs: 5 up, 2 no change, 2 down. Strongest: SP2 + NFYA. |
| `Partner Insertion - Cross-Model.ipynb` | The same 34,290 sequences, on HyenaDNA, DNABERT-2, GENA-LM and NT 2.5B. | Run. Same pattern in every model (between-model Spearman 0.78–1.00); NT 2.5B is the clearest. |
| `Function CAV - Cross-Model, Region 8.ipynb` | Fit the function direction on HyenaDNA, DNABERT-2 and GENA-LM. Does adding it to motifbreakR still improve MPRA prediction in region 8 (LDLR promoter)? | Run in Colab (this copy has no outputs). The direction exists in every model (AUC 0.74–0.78), but the MPRA gain appears only for NT-v2 500M. |
| `Function CAV - NT 2.5B, Region 8.ipynb` | Same, NT 2.5B. | Run. Highest AUC (0.80), no MPRA gain. |
| `Function CAV - Cross-Model, Region 0.ipynb` | Same as region 8, for region 0 (SORT1 enhancer). | Run. |
| `Function CAV - NT 2.5B, Region 0.ipynb` | NT 2.5B on region 0. | Run. AUC 0.79 (> NT-v2 500M 0.77), but motifbreakR + CAV Pearson 0.25 vs 0.40 for NT-v2 500M. |
| `Function CAV - NT v2 100M and 250M, Region 0.ipynb` | Model size within one family: NT-v2 100M / 250M / 500M. | **Not run yet.** |

### Follow-ups from the meeting

| notebook | what it asks | status |
|---|---|---|
| `MPRA Evaluation - Tolerant Metric and CAV Confidence.ipynb` | (A) Score motifbreakR vs motifbreakR + CAV with smoothed tracks and with hits/misses allowing ±k bp offset; (B) does the CAV's own confidence (P(bound), held-out AUC) tell where it beats motifbreakR? (C) per-occurrence calibration. CPU only, reads existing fits. | Run 2026-11-05. |
| `Pooled Regions 1 - Sites and Fits for Every MPRA Region.ipynb` | Hits, sites and function-CAV fits for all 21 MPRA regions, five models, one fit per TF shared by all regions, each model at the layer it used before. Section 1 CPU, Section 2 GPU (resumable). Writes `fcav_pool/`. | Built 2026-11-07, **not run yet**. |
| `Pooled Regions 1d - Sites from the Middle Backward (third CPU session).ipynb`, `Pooled Regions 1e - Sites from the Middle Forward (fourth CPU session).ipynb` | Two more CPU copies of 1.1–1.4: the TFs still unbuilt are split at the middle; 1d builds toward the start, 1e toward the end. Run 1.5 (main notebook) after all four sessions print `done`. | Built 2026-11-07. |
| `Pooled Regions 1c-<model> - Fits, <model> only (GPU session).ipynb` (five: nt500m, dnabert2, genalm, hyenadna, nt2p5b) | 1c with one model each, for five GPU sessions in parallel. | Built 2026-11-07. |
| `Pooled Regions 2 - MPRA Evaluation over Every Region.ipynb` | motifbreakR vs motifbreakR + CAV on every region; pooled over all / HepG2-K562 / other-cell regions with two nulls; gain vs number of peaks and vs the CAV's confidence; per-occurrence calibration over all regions. CPU only. | Built 2026-11-07, **not run yet**. |
| `Malinois MPRA - Allelic Effects vs Function CAV.ipynb` | Gosai et al. 2024 Supp. Table 2: ~330k SNVs (UKBB/GTEx), both alleles measured in HepG2, K562 and SK-N-SH. Is the allelic effect larger where the CAV says the motif is bound; does motifbreakR + CAV beat motifbreakR; is it stronger in HepG2/K562 than in SK-N-SH? NT-v2 500M directions of the pooled run, no refit. Section 1 CPU, 2 GPU, 3 any. | Built 2026-11-09, **not run yet**. |
| `Malinois MPRA 2 - Only TFs with a Good CAV (AUC Thresholds).ipynb` | The Malinois analyses (calibration, direction, within-TF interaction, T1 prediction, detection, cell contrast) repeated on TFs with held-out ChIP AUC ≥ 0.70 / 0.75 / 0.80 (thresholds fixed in advance). CPU, reads `malinois/`. | Built 2026-11-09, **not run yet**. |
| `Malinois MPRA 3 - deltaSVM Baseline and P(bound) on Top.ipynb` | A real sequence-only, non-deep-learning baseline: deltaSVM (gkm-SVM trained on ENCODE DNase peaks, one model per cell: HepG2, K562, SK-N-SH). How well it predicts the Malinois allelic effects vs motifbreakR, and whether P(bound) adds to it (CV Pearson gain, within-TF deltaSVM × P, detection AUROC, cell contrast), per CAV-AUC tier. Section 1 CPU, once per cell (three sessions can run at once, ~1–3 h each); Section 2 CPU. | Built 2026-11-15, **not run yet**. |
| `motifbreakR Diagnosis - Why It Fails, Malinois and Kircher.ipynb` | Why motifbreakR predicts MPRA effects poorly: how much of Δ is predictable (noise ceiling), whether clear effects fall where motifbreakR calls, element activity, |alleleDiff| vs effect, per-TF correlations (by ChIP source cell), the pick among several motifs; the same checks on the 21 Kircher regions, and the two datasets side by side. CPU, reads `malinois/` and `fcav_pool/`. | Built 2026-11-15, **not run yet**. |
| `Malinois MPRA 4 - Summed motifbreakR, Each Call Weighted by P(bound).ipynb` | Every motifbreakR call kept and weighted by its occurrence's P(bound), then summed per variant (main rule Σ alleleDiff × P, fixed in advance; plus within-TF percentile variants and a learned 2-weight mix), against summed motifbreakR on the same calls. Pearson / Spearman / AUROC gains with N2 nulls, bootstrap CI, HepG2/K562 − SK-N-SH contrast, per CAV-AUC tier. CPU. | Built 2026-11-15, **not run yet**. |
| `Malinois MPRA 5 - ΔP, the CAV's Own Prediction of the Mutation Effect.ipynb` | Every (variant, motif occurrence) window re-embedded with the alternative base; ΔP = P(bound | alt) − P(bound | ref) with the same NT-v2 500M L24 directions. ΣΔP over all of a variant's occurrences vs motifbreakR / summed motifbreakR / deltaSVM (incl. variants motifbreakR does not call), its CV gain on top of them per CAV-AUC tier with N2, and the HepG2/K562 − SK-N-SH contrast. Section 1 GPU (~45–60 min), Section 2 CPU. | Built 2026-11-15, **not run yet**. |

Takeaway from the meeting: the partner-insertion experiment is well controlled and its result is consistent
across models. The MPRA comparison is noisy, because each region has only one to three strong peaks, so a
single wrong motif choice decides the correlation.

## `upstream/`: where the inputs come from

These are the NT-v2 500M notebooks that produce the motif sites, the MPRA windows and the reference
function directions that every notebook above reads. They are the source of the results in the 2026-10-20
write-up. The copies here are the code, without outputs.

| notebook | writes (on Drive) |
|---|---|
| `Function CAV - Strict Labels, PWM-GC-CpG-Matched Ridge, Region 8.ipynb` | Section 1 only is used: region 8 SNVs, JASPAR hits, ChIP-matched sites → `fcav_pgc_hk/` |
| `Function CAV - Strict Labels, PWM-GC-CpG-Matched L2 Logistic, Region 8.ipynb` | NT-v2 500M function directions, region 8 → `fcav_pgc_hk_lr/` |
| `Function CAV - Strict Labels, PWM-GC-CpG-Matched L2 Logistic, Region 0.ipynb` | sites + directions, region 0 → `fcav_pgc_hk_r0/`, `fcav_pgc_hk_lr_r0/` |
| `Function CAV - Strict Labels, PWM-GC-CpG-Matched L2 Logistic, Region 20.ipynb` | sites + directions, region 20 → `fcav_pgc_hk_r20/`, `fcav_pgc_hk_lr_r20/` |
| `Hit Selection - Function CAV vs PWM, Regions 0 8 20.ipynb` | motifbreakR vs motifbreakR + CAV on the three HepG2 MPRA regions (with outputs) |

## Data

- MPRA: saturation mutagenesis of regulatory elements (Kircher et al. 2019, CAGI5). Region 0 = SORT1
  enhancer, region 8 = LDLR promoter, region 20 = F9 promoter (all HepG2).
- ChIP-seq and DNase: ENCODE, HepG2 and K562.
- Motifs: JASPAR vertebrate matrices, scanned with FIMO.
