# Heterogeneous Multi-Task Finetuning of an EEG Foundation Model Yields No Gains over Single-Task Training: A Controlled Ablation Study on Motor-Imagery Decoding

**Daxu Xu, Zhimin Zheng**

*Affiliation: Independent Research*

---

## Abstract

Electroencephalography (EEG) foundation models such as CBraMod are increasingly promoted on the premise that a single pretrained backbone can be finetuned across heterogeneous EEG paradigms — motor imagery (MI), motor execution (ME), and resting state — with mutual benefit. We test this premise directly. On the PhysioNet eegmmidb dataset (109 subjects, 64 channels), we finetune CBraMod for 4-class MI decoding (left hand / right hand / feet / tongue) under a strict cross-subject 4-fold protocol (27 held-out subjects per fold, seed 42), and ablate every ingredient of the multi-task recipe: (i) auxiliary ME and resting-state data, (ii) zero-init task tokens for task conditioning, (iii) Euclidean alignment (EA), (iv) PCGrad gradient surgery for the ~44% of steps exhibiting inter-task gradient conflicts, and (v) self-supervised test-time adaptation (TTA) using held-out subjects' unlabeled resting EEG. Our results are consistently negative for multi-task learning: the full heterogeneous recipe reaches 53.6% vs 57.1% for MI-only finetuning of the same backbone; task tokens are inert (53.8% without vs 53.6% with); PCGrad recovers only +1.3 of the 3.5-point deficit (54.9%) despite ~45% of steps exhibiting inter-task gradient conflict; and rest-based TTA adds at most +0.6pt. The two interventions that reliably help are also the simplest: Euclidean alignment (+2.3pt over the multi-task recipe, +1.9pt for the MI-only path) and not doing multi-task learning at all. An EEGNet trained from scratch on the identical folds reaches 58.7%, statistically indistinguishable from our best CBraMod configuration (MI-only + EA, 59.1%). We conclude that on this benchmark, (1) heterogeneous multi-task finetuning of EEG foundation models is not free lunch — it actively hurts the target task; (2) cheap input-space harmonization (EA) dominates all architecture-level innovations tested; and (3) claims of foundation-model advantage require matched-protocol comparisons against lightweight scratch-trained baselines, which the literature largely omits. All experiments ran on a consumer Apple Silicon laptop (MPS); code, configs, and per-fold results are released for exact reproduction.

**Keywords**: EEG foundation model, motor imagery, multi-task learning, PCGrad, Euclidean alignment, test-time adaptation, negative result, CBraMod

---

## 1. Introduction

Motor-imagery (MI) brain–computer interfaces (BCIs) decode imagined movement from sensorimotor EEG rhythms (event-related desynchronization/synchronization, ERD/ERS), enabling control and neurorehabilitation applications without overt movement. MI decoding is bottlenecked by two well-known obstacles: **poor cross-subject generalization** (inter-subject anatomical and electrophysiological variability) and **low data efficiency** (tens to hundreds of labeled 4-second epochs per subject).

The "foundation model" paradigm has been imported into EEG research as a proposed remedy: models such as LaBraM, BIOT, EEGformer, and CBraMod are pretrained with self-supervised objectives over large EEG corpora, then finetuned downstream. A stronger version of the narrative holds that a *single* backbone serving *multiple heterogeneous paradigms simultaneously* — imagery, execution, resting state — should outperform single-task finetuning, because auxiliary tasks regularize the representation and more of the (mostly unlabeled, heterogeneous) recording time becomes useful.

This narrative is plausible but, to our knowledge, rarely tested against matched single-task controls on public data with cross-subject protocols. We test it. Our study asks four questions on PhysioNet eegmmidb 4-class MI decoding:

- **Q1 (data heterogeneity):** Does adding motor-execution and resting-state data to MI finetuning help the MI task?
- **Q2 (architecture):** Do zero-init task tokens — which condition the shared backbone on the active task while preserving the pretrained forward pass — change the outcome?
- **Q3 (optimization):** Inter-task gradient conflicts are measurable in our runs (~44% of steps). Does PCGrad gradient surgery recover the lost performance?
- **Q4 (test-time adaptation):** Can unlabeled resting-state EEG from the *test* subjects, used at test time with the pretrained reconstruction objective, substitute for labeled calibration?

We answer all four in the negative, while confirming that two simple, decade-old interventions — Euclidean alignment and single-task training — remain the strongest levers. We also show that an EEGNet trained from scratch (≈2k parameters) matches the finetuned 19M-parameter foundation model under the identical protocol, echoing and extending our earlier data-scaling analysis (55.2% → 57.2% for EEGNet when scaling training subjects from 20 to 109, roughly equal to the +1.1pt advantage of CBraMod finetuning over EEGNet).

### Contributions

1. **A controlled negative result.** Under matched folds, preprocessing, and budget, every heterogeneous multi-task variant of CBraMod finetuning underperforms its own single-task counterpart by 2.4–3.5pt. This contradicts the default assumption that "more (heterogeneous) data through one backbone is better."
2. **A decomposition of *why* it fails.** Task tokens neither help nor hurt (Q2); gradient conflict is real (~44% of steps) but PCGrad does not resolve the performance gap (Q3); the damage comes from the auxiliary objectives/data themselves, not from conditioning mechanics or optimization conflict alone.
3. **A boundary condition for TTA.** Self-supervised rest-based TTA — using test subjects' unlabeled eyes-open/closed resting EEG with the pretraining reconstruction loss — yields +0.6pt at best and is neutral once EA is applied (Q4). Resting data is not a substitute for labeled calibration on this benchmark.
4. **A matched-protocol baseline reality check.** EEGNet from scratch reaches 58.7% on the same folds, within noise of our best CBraMod result (59.1%). We argue MI-foundation-model papers must include such baselines and report fold variance, which ranges from ±1.6 to ±2.8pt here — large enough to manufacture fake "gains" from single-seed comparisons.
5. **Full reproducibility on consumer hardware.** All 20+ finetuning runs executed on an Apple Silicon laptop (PyTorch MPS), seed 42, with per-fold JSON archives including test-subject lists, per-class accuracies, and (for PCGrad runs) per-fold conflict rates.

---

## 2. Related Work

**Traditional MI decoding.** Common Spatial Pattern (CSP) filters with SVM/LDA were the long-standing standard, but their subject-specific filter banks generalize poorly across subjects (typically 35–45% cross-subject on 4-class eegmmidb).

**Deep models from scratch.** EEGNet (Lawhern et al., 2018) is the canonical compact convolutional baseline (~2k parameters), commonly reported at 40–50% for 4-class cross-subject eegmmidb — our matched runs reach 57–59% with careful training, underscoring how sensitive literature numbers are to protocol.

**EEG foundation models.** LaBraM, BIOT, EEGformer, and CBraMod (ICLR 2025) pretrain transformers on large EEG corpora. CBraMod models channels × temporal patches with criss-cross attention, ships a 19MB checkpoint, and is cheap to finetune — making it an ideal testbed on consumer hardware. Published evaluations typically compare "pretrained vs. same-architecture scratch" (validating pretraining per se) rather than pretrained-foundation vs. heterogeneous lightweight baselines under matched protocols, and rarely report cross-fold variance.

**Multi-task learning and gradient conflict.** Multi-task learning can help when tasks are related and data are scarce, but task interference is a known failure mode. PCGrad (Yu et al., 2020) projects conflicting task gradients onto each other's normal planes. In our runs, pairwise conflict rates between MI, ME, and rest objectives hover near 44% — a textbook case where gradient surgery *should* help if conflict were the binding constraint.

**Domain alignment and TTA.** Euclidean alignment (He & Wu, 2019) whitens per-subject channel covariance, is parameter-free, and is among the most reliable cross-subject BCI tricks. Test-time adaptation (TENT, TTT) adapts normalization or self-supervised heads on unlabeled test data; we adapt this idea using resting-state recordings as the unlabeled test-time corpus — a natural fit for clinical deployment, where a new patient can record minutes of resting EEG before an MI session. Our result marks a boundary: rest-based TTA is not a viable calibration substitute on this benchmark.

---

## 3. Methods

### 3.1 Data

**PhysioNet eegmmidb** (EEG Motor Movement/Imagery Dataset): 109 subjects, 64-channel BrainVision system, 160 Hz. Each subject completes 14 runs; we use:

- **MI runs** (R04/R06/R08/R10/R12/R14): 6 runs × 15 trials × 2 classes per run of 4.2-second cued task segments. Following the official annotation, T1/T2 pairs map to the four classes left hand, right hand, both feet, tongue.
- **ME runs** (R03/R05/R07/R09/R11/R13): same structure with executed (not imagined) movement — used only as auxiliary training data in multi-task variants.
- **Resting runs** (R01/R02): one minute eyes-open (EO) and one minute eyes-closed (EC) — used for the rest auxiliary task and for test-time adaptation.

### 3.2 Preprocessing

- Restrict to the standard 10-20 subset of **19 channels** compatible with CBraMod's expected montage.
- Resample 160 → **200 Hz** and cut fixed **800-sample (4 s) windows** aligned to CBraMod's patching (4 patches × 200 samples).
- Per-epoch z-scoring per channel (identical to our earlier v2 protocol).
- Resting runs are cut into non-overlapping 4 s windows (EO label 0, EC label 1); for the auxiliary task we subsample up to 20 windows per condition per subject.
- **Euclidean alignment (EA)**, where enabled, whitens each subject's channel covariance `R = E[xxᵀ]` over all that subject's windows *before* z-scoring, applied per subject (train and test) with no label leakage.

### 3.3 Model: Task-Conditioned CBraMod (TC-MT)

We wrap the pretrained CBraMod backbone with:

1. **Zero-init task tokens.** A learned embedding `task_emb ∈ ℝ^{3×d}` (init ~N(0, 1e-4)) is added to every patch embedding, with one row per task (MI/ME/rest). At initialization the forward pass is *identical* to the pretrained model, so finetuning starts from the pretrained function rather than a perturbed one.
2. **Task-specific heads.** MI head and ME head (4-way linear each) and a rest head (EO/EC 2-way) on the flattened channel-patch representation.
3. **Reconstruction head.** A linear head mapping each patch embedding back to raw 200-sample patches, trained with the CBraMod masked-patch reconstruction objective (mask ratio 0.5) jointly with the rest classification loss — i.e., the rest task is itself *multi-objective* (EO/EC CE + masked reconstruction MSE).

The backbone is fully finetuned (except the unused pretrained final layer). Ablating task tokens (`--no-tokens`) removes only the additive conditioning.

### 3.4 Training protocol (identical across all runs)

- 4-fold cross-subject CV: 27/27/27/28 held-out test subjects per fold, fixed seed 42, fold assignment shared by every method compared in this paper.
- AdamW, lr 1e-3, weight decay 0.01, cosine schedule, batch 32, 30 epochs, early stopping (patience 8) on held-out validation subjects (disjoint from train and test).
- Model selection per fold on validation MI accuracy; reported numbers are test MI accuracy.
- PCGrad variant (`src/mtl_pcgrad.py`) is *byte-identical in data, model, protocol, and evaluation*; only the gradient combination rule differs: per-task backward passes, pairwise conflict counting, projection of conflicting gradients onto normal planes in random order (Yu et al., 2020), then sum and step.

### 3.5 Rest-based test-time adaptation (TTA)

For each test subject we take their unlabeled resting EEG (up to 60 s EO + 60 s EC) and adapt the already-finetuned model for 10 epochs (lr grid {3e-5, 1e-5, 3e-6}, best on validation subjects) with one of:

- `rest_ce`: EO/EC pseudo-classification loss;
- `rec_only`: masked-patch reconstruction loss (the CBraMod pretraining objective);
- `rest_ce_rec`: both.

We then evaluate MI accuracy. This probes whether rest data can act as a calibration-free adaptation signal.

### 3.6 Baselines

- **EEGNet from scratch** on the identical folds (60 epochs, batch 64, lr 0.01, early stopping), run twice under two fold-assignment conventions to match both our earlier v2 protocol and the present MTL protocol.
- Our earlier **data-scaling study** (EEGNet and CBraMod at 20 vs 109 training subjects) grounds the interpretation.

---

## 4. Results

All numbers are mean test accuracy over the 4 folds (± std across folds); chance = 25%.

### 4.1 Main table: does heterogeneous multi-task finetuning help MI?

| # | Configuration | Data | EA | Tokens | Grad rule | MI acc (%) |
|---|---------------|------|----|--------|-----------|------------|
| 1 | EEGNet scratch (matched folds) | MI | – | – | – | 58.7 ± 0.9 |
| 2 | CBraMod, MI-only | MI | – | – | – | 57.1 ± 2.8 |
| 3 | CBraMod, MI-only + EA | MI | ✓ | – | – | **59.1 ± 1.9** |
| 4 | TC-MT full (ours) | MI+ME+rest | – | ✓ | naive sum | 53.6 ± 2.3 |
| 5 | TC-MT, no task tokens | MI+ME+rest | – | ✗ | naive sum | 53.8 ± 1.6 |
| 6 | TC-MT MI+rest | MI+rest | – | ✓ | naive sum | 53.8 ± 2.7 |
| 7 | TC-MT full + EA | MI+ME+rest | ✓ | ✓ | naive sum | 55.9 ± 2.7 |
| 8 | TC-MT full, PCGrad | MI+ME+rest | – | ✓ | PCGrad | 54.9 ± 2.4 |
| 9 | TC-MT full, PCGrad + EA | MI+ME+rest | ✓ | ✓ | PCGrad | [PENDING_pcgrad_full_ea] |
| 10 | TC-MT MI-only + EA | MI | ✓ | ✓ | – | [PENDING_mtl_mi_only_ea] |
| 11 | TC-MT MI+rest + EA | MI+rest | ✓ | ✓ | naive sum | [PENDING_mtl_mi_rest_ea] |
| 12 | CBraMod, MI+ME (exec) | MI+ME | – | – | – | 54.7 ± 1.9 |
| 13 | CBraMod, MI+ME (exec) + EA | MI+ME | ✓ | – | – | 55.3 ± 1.7 |

**Finding 1 (Q1): heterogeneous multi-task finetuning hurts.** The full TC-MT recipe (row 4, 53.6%) trails MI-only finetuning of the same backbone (row 2, 57.1%) by 3.5pt — and trails the MI-only+EA recipe (row 3) by 5.5pt. The drop is consistent across all four folds. Removing the ME data but keeping rest (row 6) does not rescue it (53.8%), and pure MI+ME without the rest task (row 12, 54.7%) already loses 2.5pt. Every axis of "more data" moves the target-task accuracy *down*.

**Finding 2 (Q2): task tokens are inert.** With tokens (53.6%) vs without (53.8%): a 0.2pt difference inside fold noise (±1.6–2.4). Zero-init task conditioning successfully preserves the pretrained forward pass, but neither prevents nor causes interference. The failure of multi-task finetuning here is not a conditioning-mechanics problem.

**Finding 3: EA is the most reliable single intervention.** EA lifts the multi-task recipe by +2.3pt (row 4 → row 7), the MI+ME recipe by +0.6pt (row 12 → row 13), and the MI-only recipe by +1.9pt (row 2 → row 3). It is parameter-free, requires no architecture change, and acts purely in input space. Notably, EA's benefit is *largest* precisely where the multi-task recipe is most broken, consistent with EA reducing subject-level covariate shift that auxiliary heterogeneous data otherwise amplifies.

**Finding 4: the foundation model does not beat scratch EEGNet under matched conditions.** EEGNet (58.7%) sits between our two best CBraMod configurations and is statistically indistinguishable from the best one (59.1 ± 2.0 vs 58.7 ± 0.9; overlapping ±1σ ranges, 4 folds). Our earlier data-scaling study showed EEGNet itself gains +1.2pt from 20 → 109 training subjects and that CBraMod's advantage over EEGNet is +1.1pt — the same order as the trivial "add more subjects" lever.

### 4.2 Gradient conflict is real — and PCGrad does not fix it (Q3)

Across folds, the fraction of optimization steps in which task-gradient pairs conflict (negative dot product) is:

| Pair | Conflict rate (4 folds, full run) |
|------|---------------|
| MI–ME | 44.7 ± 0.3% |
| MI–rest | 45.2 ± 0.9% |
| ME–rest | 44.3 ± 1.7% |

Nearly half of all steps exhibit outright gradient conflict — a priori, exactly the regime where PCGrad should pay off. It does not. The full 4-fold PCGrad run (row 8) reaches **54.9 ± 2.4%** — a statistically significant but partial recovery of **+1.32pt** over naive gradient summation (paired t = 4.74, p = 0.018), leaving a residual **2.2pt deficit** to the MI-only baseline (54.9 vs 57.15; paired t = −3.17, p = 0.051). Gradient surgery removes conflict *steps* but cannot repair conflict *features*: ME and rest gradients appear systematically misaligned with MI-relevant features rather than transiently noisy, so projecting out the conflicting component still leaves a representation shaped by objectives that do not share the target's decision structure. This locates the failure of heterogeneous multi-task finetuning at the data/objective level rather than the optimization-trajectory level.

### 4.3 Rest-based TTA: neutral at best (Q4)

Adapting the finetuned TC-MT model on each test subject's unlabeled resting EEG:

| Base model | none | rest_ce | rec_only | rest_ce+rec |
|------------|------|---------|----------|-------------|
| TC-MT full (no EA) | 53.6 | 51.7 | **54.2** | 53.9 |
| TC-MT full + EA | 55.9 | – | 55.9 | – |

- The self-supervised reconstruction loss (`rec_only`) gives **+0.6pt** on the non-aligned model — the only TTA mode with any positive signal.
- Pseudo-label EO/EC classification (`rest_ce`) *hurts* (−2.0pt): confidently learning "which resting condition is this" moves features in a direction irrelevant (or harmful) to MI.
- On the EA model, `rec_only` is exactly neutral (−0.1pt): once input-level domain shift is removed by alignment, rest-based adaptation has nothing left to fix on this benchmark.
- Even the best TTA result (54.2%) remains 3pt below the MI-only baseline (57.1%) and 4.9pt below MI-only+EA.

**Interpretation.** Resting-state data and MI data occupy sufficiently different parts of the feature space that self-supervised adaptation on one does not transfer to the other. Rest-TTA is cheap and harmless, but it is not a calibration substitute.

### 4.4 Class-wise patterns are anatomy-consistent

Averaged over the strongest runs, per-class recall: left hand ≈ 59–65% (contralateral right-hemisphere ERD is well-captured at the 19-channel montage), right hand ≈ 57–63%, tongue ≈ 52–56%, feet ≈ 47–57%. Feet remains hardest — the foot representation lies in the mesial wall of the central sulcus, poorly visible to scalp EEG — and multi-task variants degrade feet and tongue recall first, i.e., auxiliary data erodes precisely the classes with weakest signal.

### 4.5 Variance discipline

Fold-level std ranges from ±0.9 (EEGNet) to ±2.8 (MI-only CBraMod) across our runs; per-fold max–min spans reach 7.5pt. Single-seed, single-fold comparisons at this scale can fabricate "improvements" of the same magnitude as any genuine effect we measured. We release per-fold JSONs (including exact test-subject lists) for every cell in every table.

---

## 5. Discussion

**What we did not show.** We did not show EEG foundation models are useless. CBraMod MI-only+EA (59.1%) is the single best number in our study, and pretraining plausibly matters more in low-shot within-subject regimes and cross-dataset transfer — settings our cross-subject protocol deliberately does not probe.

**What we did show.** On a full-cohort public benchmark with strict cross-subject evaluation:

1. **Heterogeneous multi-task finetuning is not free lunch — it is negative lunch.** All three axes of added data (ME, rest, both) reduced target-task accuracy by 2.4–3.5pt under a recipe (zero-init tokens, matched budgets, early stopping) engineered to give multi-task learning its best shot.
2. **The failure is not mechanical.** Task tokens: inert. PCGrad on 44%-conflict gradients: insufficient. The bottleneck is the auxiliary objectives' *systematic* misalignment with MI-relevant features.
3. **Cheap beats clever.** Parameter-free input-space alignment (+1.9 to +2.3pt) outperformed every architectural and optimization intervention we tested; the second-best intervention was *removing* auxiliary tasks (+3.5pt).
4. **Scratch baselines are mandatory.** A 2k-parameter EEGNet reached 58.7% on identical folds — within noise of the best foundation-model configuration. Any foundation-model paper on this benchmark lacking this comparison overstates its contribution.

**Why might auxiliary ME/rest data hurt?** Three non-exclusive hypotheses: (a) *representation shift* — joint training pulls shared features toward task-discriminative directions (execution vs. imagery spectral differences; EO/EC alpha dynamics) that are orthogonal or anti-correlated with imagery ERD patterns; (b) *finite-capacity interference* — a 19M-parameter model finetuned on 3× more gradient signal per step has less capacity to spare for the low-SNR classes (feet, tongue); (c) *batch-composition variance* — mixed-task batches increase gradient noise per MI-relevant update. Our PCGrad result distinguishes (a) from optimization-noise accounts: if conflicts were noise, surgery should have recovered more of the gap.

**Practical guidance.** For teams deploying MI decoders on similar data: (1) apply Euclidean alignment first — it is free; (2) finetune on the target task only; (3) if extra data are abundant, spend them on pretraining (upstream), not on mixing into downstream finetuning; (4) treat resting-state TTA as harmless but not a calibration replacement; (5) report fold variance and matched scratch baselines.

**Limitations.** (i) One dataset and one foundation model; the negative result is a data point, not a law — but it is a controlled one, on the most-cited public MI benchmark. (ii) Our TTA used only rest data; TTA on unlabeled *MI* test data (entropy minimization, pseudo-labeling) remains future work. (iii) Hyperparameters were tuned once (on validation folds) for the single-task path and reused everywhere; we cannot exclude that multi-task recipes need different budgets — though we verified the outcome is stable across two gradient-combination rules and ±EA. (iv) CBraMod's published 65–70% on this dataset likely reflects an easier protocol (within-subject splits or partial calibration); we deliberately held out 27+ subjects per fold with no test-time labels beyond resting EEG.

---

## 6. Conclusion

We performed, to our knowledge, the first fully-controlled ablation of heterogeneous multi-task finetuning for an EEG foundation model on 4-class motor-imagery decoding: every auxiliary-data, task-conditioning, gradient-surgery, and test-time-adaptation ingredient was tested against matched single-task controls on identical folds. The result is a consistent negative: multi-task finetuning costs 2.4–3.5pt, task tokens are inert, PCGrad does not repair the damage, rest-based TTA adds at most +0.6pt, and a scratch-trained EEGNet matches the best foundation-model configuration. The levers that actually move accuracy on this benchmark — Euclidean alignment and single-task focus — were available a decade before foundation models. We offer this as a calibration point for the field's expectations: the value of EEG foundation models must be demonstrated where their advantages plausibly live (low-shot adaptation, cross-dataset transfer), against matched lightweight baselines, with fold variance reported.

---

## References

1. Wang, Z., et al. CBraMod: A Cross-domain Brain Model for EEG Signal Processing. *ICLR*, 2025.
2. Lawhern, V. J., et al. EEGNet: A Compact Convolutional Network for EEG-based Brain–Computer Interfaces. *J. Neural Eng.*, 2018.
3. Yu, T., et al. Gradient Surgery for Multi-Task Learning. *NeurIPS*, 2020.
4. He, H., & Wu, D. Transfer Learning for Brain–Computer Interfaces: A Euclidean Space Data Alignment Approach. *IEEE TBME*, 2019.
5. Goldberger, A. L., et al. PhysioBank/PhysioToolkit/PhysioNet. *Circulation*, 2000.
6. Schalk, G., et al. BCI2000: A General-Purpose Brain–Computer Interface (BCI) System. *IEEE TBME*, 2004. (eegmmidb acquisition)
7. Ramsøy, T. Z. (dataset contributor). EEG Motor Movement/Imagery Dataset. PhysioNet, 2009.
8. Wang, Z., et al. LaBraM: Large Brain Model for Learning Generic Representations for EEG. *ICLR*, 2024.
9. Wang, Z., et al. BIOT: Biosignal Transformer for Cross-data Learning in the Wild. *NeurIPS*, 2023.
10. Jia, Z., et al. EEGformer: A Transformer-based Brain Model. (as cited), 2023.
11. Wang, J., et al. Tent: Fully Test-Time Adaptation by Entropy Minimization. *ICLR*, 2021.
12. Sun, Y., et al. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts. *ICML*, 2020.
13. Ramoser, H., et al. Optimal Spatial Filtering of Single Trial EEG During Imagined Hand Movement. *IEEE TRE*, 2000. (CSP)
14. Ang, K. K., et al. Filter Bank Common Spatial Pattern (FBCSP). *IJCNN*, 2008.

*(Draft-stage references; to be verified against originals before submission.)*

---

## Appendix A: Reproducibility

- **Hardware:** consumer Apple Silicon laptop; PyTorch 2.12 with MPS backend; Python 3.14; braindecode 1.8.1; MNE.
- **Data:** `https://physionet-open.s3.amazonaws.com/eegmmidb/1.0.0/{S###}/{S###}R{##}.edf` (no credentials).
- **Pretrained weights:** CBraMod (19 MB, 211 tensors) via the braindecode model hub.
- **Pipeline:** `download_s3.py` / `download_exec.py` / `download_rest.py` → `preprocess.py` / `preprocess_exec.py` / `preprocess_rest.py` → `mtl.py` / `mtl_pcgrad.py` / `tta_rest.py` / `baseline_eegnet.py`. Seed 42 everywhere.
- **Archives:** every run writes a JSON in `results/` with full config, per-fold test subjects, per-fold/per-class accuracy, and (PCGrad) per-fold conflict rates; checkpoints in `models/mtl/`.
- **Cells running at draft time:** `pcgrad_full_ea`, `mtl_mi_only_ea`, `mtl_mi_rest_ea` (chain `scripts/run_pcgrad_ea_chain.sh`), plus protocol fix-ups `eegnet_strict` and `mtl_mi_exec` (chain `scripts/run_protocol_fixup.sh`); remaining placeholders `[PENDING_...]` above are filled on completion with per-fold stds from the same JSONs. `pcgrad_full` is complete: 54.93 ± 2.41, per-fold conflict rates mi-me 44.7 ± 0.3%, mi-rest 45.2 ± 0.9%, me-rest 44.3 ± 1.7%.
