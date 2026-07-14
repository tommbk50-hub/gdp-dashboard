# 🧬 XGBoost-GPU Protein Pocket Predictor

An end-to-end, **GPU-native machine learning pipeline** for detecting, featurizing, and ranking druggable ligand-binding pockets on protein surfaces. It is a modern, highly parallelized reimplementation of the ideas behind the classical template-free predictor **[P2Rank](https://github.com/rdk/p2rank)**, rebuilt from the ground up so that the mathematically intensive spatial work runs **natively on an NVIDIA GPU** via CuPy, cuML, and hand-written C++/CUDA kernels.

<img width="690" height="505" alt="image" src="https://github.com/user-attachments/assets/ac49024e-4e8c-4f65-b6d0-9f84535d0cce" />

<img width="636" height="557" alt="image" src="https://github.com/user-attachments/assets/a2e93575-5eda-42b0-8d88-5761ed4a59f1" />

The current, authoritative workflow lives in the Jupyter/Colab notebook **`scPDB_Data_version_7_XGBoost_GPU.ipynb`**. This README uses that notebook as its single source of truth and explains, section by section, **how the model works, how it is trained on real experimental data, and how data leakage is rigorously prevented** — plus how the same design scales up to millions of proteins by keeping data resident in GPU VRAM.

> ✅ **This is no longer a mock-target proof of concept.** Earlier versions trained on a *randomly generated* target vector. Version 7 trains a **single universal pocket predictor on real ground truth** derived from the **scPDB** database of ~16,000 experimentally determined protein–ligand complexes. The label for every surface point comes from the *actual co-crystallized ligand* in each complex, and the training/test split is protected by a strict, homology-aware data-leakage firewall.

---

## Table of Contents

1. [Motivation: Why GPU-Native Pocket Prediction?](#-motivation-why-gpu-native-pocket-prediction)
2. [How This Builds on P2Rank](#-how-this-builds-on-p2rank-random-forest--xgboost-gpu)
3. [Pipeline at a Glance](#-pipeline-at-a-glance)
4. [How the Model Works — Detailed Walkthrough](#-how-the-model-works--detailed-walkthrough)
5. [How Real Data Trains the Model](#-how-real-data-trains-the-model)
6. [How Data Leakage Is Prevented](#-how-data-leakage-is-prevented)
7. [Feature Set (the 10-Dimensional Surface Descriptor)](#-feature-set-the-10-dimensional-surface-descriptor)
8. [Why XGBoost-GPU Instead of Random Forest](#-why-xgboost-gpu-instead-of-random-forest)
9. [Evaluation & Results](#-evaluation--results)
10. [Scaling to the Proteome (Millions of Proteins)](#-scaling-to-the-proteome-millions-of-proteins)
11. [Limitations & Path to a Production-Ready Model](#-limitations--path-to-a-production-ready-model)
12. [Project Evolution & Version History](#-project-evolution--version-history)
13. [Environment & Dependencies](#-environment--dependencies)
14. [How to Run](#-how-to-run)
15. [Output Files](#-output-files)

---

## 🎯 Motivation: Why GPU-Native Pocket Prediction?

Identifying where a small molecule can bind on a protein's surface is the first step of structure-based drug discovery. Classical tools such as P2Rank are accurate but **CPU-bound**: they were designed to analyze proteins one (or a handful) at a time. Modern structural biology has flipped the bottleneck — with the AlphaFold Protein Structure Database now containing **hundreds of millions of predicted structures**, the limiting factor is no longer *whether* we have a structure, but *how fast* we can mine all of them for druggable sites.

This pipeline is built around one central engineering principle: **never leave the GPU.** From the moment the atomic coordinates are parsed, every heavy computation — spatial hashing, surface simulation, pharmacophore perception, model training, model inference, and clustering — happens on VRAM-resident arrays. There is **no PCI-e ping-pong** between CPU and GPU inside the hot path. This "zero-copy" design is what makes it feasible to train on the whole scPDB database and, eventually, to screen the whole proteome rather than a single crystal structure.

---

## 🔬 How This Builds on P2Rank (Random Forest → XGBoost-GPU)

P2Rank is a **template-free, machine-learning** method for ligand-binding-site prediction. Its core algorithm is:

1. Generate a cloud of points on the **Solvent-Accessible Surface (SAS)** of the protein.
2. For each SAS point, compute a **local feature vector** describing the surrounding chemistry (atom types, physicochemical properties) within a distance cutoff.
3. Score each point with a **Random Forest** classifier trained on known protein–ligand complexes.
4. **Cluster** the high-scoring points into discrete pockets and rank them.

This notebook mirrors that exact logic, but re-engineers every stage for the GPU and modernizes the key components:

| Stage | P2Rank (classic) | This project (version 7) |
|-------|------------------|--------------------------|
| Training data | Curated protein–ligand complexes | **scPDB** (~16,000 complexes) streamed folder-by-folder |
| Ground-truth labels | Known binding residues | SAS points within **2.5 Å** of the real co-crystallized `ligand.mol2`, via CPU `cKDTree` |
| SAS representation | Connolly/SAS points (CPU) | 500,000 random sample points, steric-clash rejection inside a CUDA kernel |
| Local featurization | Per-point neighborhood features (CPU) | Custom CUDA `aggregate_features` kernel over a spatial-hash grid |
| Scoring model | **Random Forest (CPU)** | **XGBoost regressor** (`tree_method='hist'`, `device='cuda'`) |
| Class imbalance | — | Dynamic `scale_pos_weight` (≈ 64) per training chunk |
| Clustering | Custom hierarchical clustering | cuML **HDBSCAN** on the GPU |
| Data locality | CPU arrays | Everything resident in **VRAM (CuPy)** |

The most important substitution is **Random Forest → XGBoost-GPU**. P2Rank's Random Forest builds many independent trees on the CPU and averages them. XGBoost instead builds trees *sequentially* (gradient boosting), where each new tree corrects the residual error of the previous ensemble, and — crucially — its histogram algorithm (`tree_method='hist'`) was purpose-built for GPU acceleration. Because our features are already CuPy arrays living in VRAM, XGBoost consumes them **zero-copy**, completely avoiding the CPU↔GPU transfer bottleneck that would otherwise dominate at scale.

---

## 🗺️ Pipeline at a Glance

The notebook is organized into five phases:

```
 ┌── PHASE 0-2: TRAINING (real scPDB data) ─────────────────────────────┐
 │                                                                      │
 │  scPDB database (~16,000 protein–ligand complexes)                   │
 │        │  each folder = protein.mol2 (A) + ligand.mol2 (B)           │
 │        ▼                                                             │
 │  [Firewall] reserve 101 strict holdout base IDs (incl. 7t2t)         │
 │        │                                                             │
 │        ▼                                                             │
 │  [Audit] UniProt homology check — ban homologs from training        │
 │        │                                                             │
 │        ▼                                                             │
 │  process_folder(): native mol2 parse → GPU spatial hash →            │
 │        │           SAS + CUDA feature extraction →                   │
 │        │           cKDTree ground-truth labelling vs ligand.mol2     │
 │        ▼                                                             │
 │  Out-of-core: fuse folders into X_chunk_*.npy / Y_chunk_*.npy        │
 │        │                                                             │
 │        ▼                                                             │
 │  Incremental GPU XGBoost training (warm-started per chunk,           │
 │        │  scale_pos_weight ≈ 64.2 for the ~1.5% positive rate)       │
 │        ▼                                                             │
 │  data_leak_audited_universal_scPDB_pocket_weights.pkl                │
 └──────────────────────────────────────────────────────────────────────┘
             │
             ▼
 ┌── PHASE 3: INFERENCE on an unseen target (Apo SARS-CoV-2 MPro, 7T2T) ─┐
 │  fetch PDB → clean → PDB2PQR (pH 7.4, AMBER) → RDKit-free parser →    │
 │  GPU spatial hash → SAS + features → XGBoost score → HDBSCAN →       │
 │  rank pockets → export JSON → Pocket 1 pharmacophore map             │
 └──────────────────────────────────────────────────────────────────────┘
             │
             ▼
 ┌── PHASE 4-5: EVALUATION ─────────────────────────────────────────────┐
 │  fragment coverage vs real XRD fragments, Pearson correlations,      │
 │  runtime analytics, per-target UniProt leakage re-audit, and a       │
 │  large-scale run over the full 422-protein holdout set (PR-AUC)      │
 └──────────────────────────────────────────────────────────────────────┘
```

---

## 🔍 How the Model Works — Detailed Walkthrough

The model is a **gradient-boosted decision-tree regressor** that assigns every point on a protein's solvent-accessible surface a **ligandability score** between 0 and 1. Those scores are then clustered into discrete, ranked pockets. The full path from atoms to pockets is below.

### 1. Native Mol2 / PQR parsing → GPU VRAM (RDKit-free)
For **training**, each scPDB complex ships as two Tripos `.mol2` files — `protein.mol2` (Structure A, the target we featurize) and `ligand.mol2` (Structure B, the ground-truth binder). A pure-text parser (`parse_mol2_structure_a`) reads only the `@<TRIPOS>ATOM` block and emits flat NumPy arrays of **coordinates, partial charges, and element IDs**. Element identity is taken from the unambiguous SYBYL atom type (e.g. `C.ar`, `N.4`, `Zn`), deliberately avoiding a greedy two-character read of PDB-style atom names (so `CA`/`CD`/`CE` stay carbons, not calcium/cadmium/cerium). For **inference**, the same information comes from a `.pqr` file produced by PDB2PQR (see Phase 3). Either way, no RDKit topology perception runs in the numerical hot path.

Geometric **pharmacophore pre-tagging** (`_derive_pharmacophore_tags`) then flags each atom as donor / acceptor / hydrophobe / aromatic / positive-ionizable / negative-ionizable / Zn-binder, so the CUDA kernel later reads a boolean instead of re-deriving chemistry per point. All arrays are pushed to VRAM with `cp.asarray(...)`; from here the protein lives entirely as synchronized CuPy arrays.

### 2. GPU spatial hashing & grid construction
Computing the distance from every SAS point to every atom is `O(N²)` and intractable at scale. To make neighbor lookups `O(1)`, a **uniform 4.0 Å spatial-hash grid** is fitted to the protein and two hand-written C++/CUDA kernels are compiled once via `cp.RawModule`:
- `calc_hash` — assigns each atom a flat cell hash `cx + cy·Dx + cz·Dx·Dy`.
- `find_cell_bounds` — after a GPU `argsort` of the hashes, records the start/end index of each occupied cell.

Every coupled physics array (charges, elements, pharmacophore tags) is reordered by the *same* sort permutation so it stays perfectly aligned with the grid.

### 3. SAS generation + native geometric feature extraction
This is the pharmacophore-perception heart of the pipeline:
- **500,000** candidate surface points are sampled uniformly inside the protein bounding box.
- A single custom CUDA kernel, `aggregate_features`, runs one thread per SAS point. Each thread visits only its **3×3×3 neighborhood of grid cells** and, for atoms within the **4.0 Å cutoff**, accumulates distance-weighted feature densities.
- **Steric-clash rejection** simulates a true Solvent-Accessible Surface: if any protein atom is within the **2.5 Å steric cutoff**, the point is flagged as *inside* the protein and its features are zeroed. Only points in the thin solvent-accessible shell survive.
- **Buriedness** is the local atom count divided by the 4 Å sphere volume (atoms/Å³) — flat surface points fill only ~half their sphere, deep-cavity points fill almost all of it.

The kernel does **native geometric pharmacophore perception**: it reads the pre-tagged flags directly, with **no RDKit SMARTS matching anywhere in the hot path**.

### 4. Ground-truth labelling (training only)
For every valid SAS point (those that actually intersect the protein, `f_counts > 0`), the coordinates are returned to the original frame and compared against the real ligand atoms using a CPU `scipy.spatial.cKDTree`. Any surface point within **2.5 Å (`CONTACT_CUTOFF`)** of a `ligand.mol2` atom is labelled **1.0 (true binding hotspot)**; everything else is **0.0**. This turns each complex's `(X_valid_gpu, Y_valid_gpu)` into supervised training data grounded in an *actual experimental binding event*.

### 5. XGBoost-GPU training
The 10-D feature matrix and label vector are fed to `xgb.XGBRegressor(tree_method='hist', device='cuda')`, which builds boosted trees entirely on the GPU (`learning_rate=0.05`, `n_estimators=300`, `max_depth=8`). Because training runs out-of-core over the whole database, the booster is **warm-started chunk by chunk** and the final universal weights are pickled. (Full detail in [How Real Data Trains the Model](#-how-real-data-trains-the-model).)

### 6. XGBoost-GPU inference
The trained model predicts a **ligandability score** for every point in a novel protein's feature matrix, keeping the output as a CuPy array so downstream clustering never leaves VRAM.

### 7. Pocket clustering, segmentation & ranking
- The **top 10 %** of scoring points (90th-percentile threshold) that also touch the protein are retained.
- Survivors are shifted back into true PDB coordinate space and clustered with **cuML `HDBSCAN`** on the GPU.
- Each cluster becomes a candidate pocket; its enclosed volume is measured with a 3-D **alpha-shape (concave hull)** via `alphashape`/`trimesh`, and clusters that are too small or too sparse are rejected.
- Remaining pockets are ranked by **cumulative point probability** (matching P2Rank's ranking philosophy).

### 8. Export & Pocket 1 pharmacophore mapping
The ranked pockets are written to JSON (P2Rank-style: rank, score, centroid, volume), the binding-site residues of the top pockets are saved, and everything is rendered with `py3Dmol`. Finally, for the **top-ranked pocket** the pipeline classifies each pocket atom into a pharmacophore family, producing an *idealized 3-D pharmacophore blueprint* (`[pocket geometry] → [complementary pharmacophore]`) — exactly the paired data needed to drive downstream virtual screening or generative drug-design models.

---

## 🧬 How Real Data Trains the Model

Version 7's defining change is that the model learns from **real experimental structures**, not a synthetic target.

### The dataset: scPDB
Training uses the **scPDB** database — a curated collection of **~16,000 druggable protein–ligand complexes** derived from the Protein Data Bank. The notebook downloads and extracts `scPDB.tar.gz`, giving one subfolder per complex. Each folder contains:
- `protein.mol2` — **Structure A**, the protein target that gets featurized.
- `ligand.mol2` — **Structure B**, the experimentally co-crystallized binder that provides the **ground truth**.

After the leakage firewall (below) excludes the reserved holdout, the batch run reports **16,191 unique training PDB IDs** ingested.

### Turning structures into supervised labels
Ground truth is *not* hand-annotated — it is read directly from where the ligand actually sits:

1. `process_folder()` featurizes `protein.mol2` into the 500k-point / 10-feature SAS matrix (Steps 1–3 above).
2. **Step 3.5 — cKDTree ground-truth labelling.** A `cKDTree` is built over the real `ligand.mol2` atom coordinates. Every valid SAS point within **2.5 Å** of a ligand atom is labelled `1.0`; all others `0.0`. The label therefore encodes *"a real drug-like molecule was experimentally observed binding here."*

### Handling extreme class imbalance
Only **~1.5 %** of surface points are true binding hotspots — for every positive there are ~64 negatives. Left uncorrected, a model could score ~98 % accuracy by always predicting "background" and never find a pocket. The pipeline counters this by computing, **per chunk**, `pos_weight = num_negatives / num_positives` (≈ **64.2**) and passing it to XGBoost's `scale_pos_weight`. This multiplies the gradient penalty for missing a true hotspot ~64× harder than for a background mistake, forcing the trees to actively learn the rare positives.

### Out-of-core, VRAM-managed training at scale
Featurizing all ~16k complexes at once would materialize ~39 GB of dense matrices and exhaust host RAM/VRAM. The pipeline therefore:
- Streams folders in **chunks**, fusing each chunk's features/labels and flushing them to disk as `X_chunk_*.npy` / `Y_chunk_*.npy`.
- Aggressively reclaims memory after every folder and chunk (`gc.collect()` + `cp.get_default_memory_pool().free_all_blocks()`), keeping per-folder VRAM flat.
- **Incrementally trains** one booster across the chunks, **warm-starting** from the existing model (`xgb_model=model.get_booster()`) so trees accumulate rather than retrain from scratch, and recomputing `scale_pos_weight` for each chunk's local imbalance.

The final universal model is serialized to **`data_leak_audited_universal_scPDB_pocket_weights.pkl`** (also archived as a `.tar.gz`) and reused for all downstream inference.

---

## 🛡️ How Data Leakage Is Prevented

A universal predictor is only meaningful if it has genuinely **never seen** the proteins it is later tested on — including close homologs. Version 7 enforces this with a two-layer firewall applied *before any featurization runs*.

### Layer 1 — Strict holdout set (folder-name firewall)
Before ingestion, the notebook carves out a fixed **holdout set of 101 unique 4-character base IDs**: the mandatory target **`7t2t`** (apo SARS-CoV-2 main protease) plus **100 randomly drawn** base IDs. scPDB folders are named `<base_id>_<index>` (e.g. `abcd_1`), so during the batch loop **any folder whose `folder[:4]` base ID is in the holdout set is permanently skipped**. The split is serialized to `holdout_test_set.json` so it is **reproducible across sessions** and reused for validation.

### Layer 2 — UniProt homology audit
Excluding exact PDB IDs is not enough — the *same protein* appears under many different PDB IDs. The audit cell closes this gap:

1. For each holdout base ID, query the RCSB API for its **UniProt accession** (e.g. `P0DTD1` for MPro).
2. Query RCSB again for **every PDB entry worldwide that maps to that same UniProt ID** (all homologous/identical structures).
3. Intersect that global set with the actual training PDB IDs. **Any intersection is flagged as leakage**, and the offending IDs are appended to `holdout_test_set.json` — after which the model must be retrained so those homologs are also banned.

When run against the training set, the audit confirms a **strictly zero intersection** (e.g. for MPro's `P0DTD1`, 10 global structures vs. 16,191 training IDs → 0 overlap), certifying the model was **completely blind** to the evaluation targets.

### Why this matters
Together these layers guarantee the reported zero-shot metrics reflect **true generalization**, not memorization. The saved weights are deliberately named `data_leak_audited_universal_scPDB_pocket_weights.pkl` to make the guarantee explicit, and the audit is re-run for every new inference target (e.g. MPro, and target `1AOE`/`P22906` in the large-scale test).

---

## 🧪 Feature Set (the 10-Dimensional Surface Descriptor)

Each SAS point is described by a 10-element vector computed entirely on the GPU:

| # | Feature (`X_gpu` column) | Meaning |
|---|--------------------------|---------|
| 1 | `f_counts` | Local atom neighbor count within 4 Å |
| 2 | `f_electro` | Coulombic electrostatic potential from AMBER charges |
| 3 | `f_acceptor` | H-bond acceptor density (distance-weighted) |
| 4 | `f_donor` | H-bond donor density |
| 5 | `f_hydrophobe` | Hydrophobe density |
| 6 | `f_aromatic` | Aromatic ring density |
| 7 | `f_pos_ion` | Positive-ionizable density |
| 8 | `f_neg_ion` | Negative-ionizable density |
| 9 | `f_zn_binder` | Zinc-binder density |
| 10 | `f_buriedness` | Local atom density (atoms/Å³) — cavity enclosure proxy |

---

## ⚡ Why XGBoost-GPU Instead of Random Forest

- **Ensemble strategy.** Random Forest (P2Rank's baseline) builds many *independent* trees and averages them. XGBoost builds trees *sequentially* via gradient boosting, so each tree explicitly corrects the residual errors of the previous ensemble — typically higher accuracy on complex tabular data.
- **Hardware efficiency.** Random Forests are usually CPU-bound and scale roughly linearly. XGBoost's `tree_method='hist'` bins continuous features into histograms so thousands of CUDA cores can evaluate candidate split points almost instantaneously.
- **Zero-copy VRAM execution.** Because the feature matrix is already a CuPy array in VRAM, XGBoost trains and infers without ever transferring data back to the CPU — eliminating the PCI-e bottleneck that dominates CPU-based pipelines at scale.

---

## 📊 Evaluation & Results

The notebook validates the universal model on the strictly withheld **apo SARS-CoV-2 main protease (PDB `7T2T`)** and on the full 422-protein holdout set.

- **Fragment coverage.** Predicted pockets are checked against **1,240 real X-ray crystallographic fragment coordinates** (within an 8 Å radius of each pocket centroid). The predicted pockets collectively enclose **1,238 / 1,240 fragments (99.8 %)**; the top-5 and top-10 ranked pockets capture **1.9 %** and **2.2 %** respectively — i.e. the geometry engine finds essentially all fragment sites, while *ranking* the single most druggable one remains the harder, open problem.
- **Statistical analysis.** Pearson correlation between predicted pocket score and fragment count is weak/negative (**r ≈ −0.03**, p ≈ 0.75), as is pocket volume vs. fragment count (**r ≈ −0.06**, p ≈ 0.50) — an honest signal that the ranking function still needs work even though pocket *localization* is strong.
- **Large-scale holdout test.** Running inference across the **422-protein holdout set** processes hundreds of models at **~106 ms per model** (total ≈ 47 s), reporting per-target **PR-AUC (average precision)** for the rare positive class.
- **Runtime analytics.** For a single target, total notebook execution (including download and biophysical prep) is ~19 s, of which the **core GPU calculation is only ~4.6 s** — demonstrating the value of keeping the whole pipeline in VRAM.
- **Leakage re-audit.** The UniProt audit is re-run for each evaluation target and confirms **zero overlap** with the training set.

> These numbers are reported *as produced by the notebook*, including the weak ranking correlations. The pocket-localization ("find all sites") result is strong; the pocket-ranking ("which site is best") result is the primary avenue for future improvement.

---

## 🚀 Scaling to the Proteome (Millions of Proteins)

The GPU-native design exists so this can run not on one protein, but on the **entire AlphaFold database**. The intended scale-up strategy:

1. **High-throughput pocket prediction.** Because spatial hashing, featurization, scoring, and clustering all run natively on the GPU, the pipeline can be deployed across millions of predicted structures, producing a massive database of binding pockets with full biophysical/geometric surface descriptions.
2. **Ground-truth pharmacophore dataset.** For every predicted pocket, extract the complementary pharmacophore (as done for Pocket 1), yielding a huge paired dataset of `[3D pocket geometry] → [ideal 3D ligand pharmacophore]`.
3. **Train an E(3)-equivariant graph neural network.** With millions of paired examples, replace the hand-engineered features with an **E(3)-equivariant GNN** that ingests raw 3-D atomic coordinates and natively respects translation/rotation/reflection symmetry — learning *directional* biophysics (e.g. H-bond angles) that scalar-distance features cannot capture.
4. **Zero-shot de novo design.** The trained network can then predict the ideal 3-D pharmacophore for a completely novel target and feed billion-compound virtual screens (e.g. ZINC22) or diffusion-based generative models.

**Engineering constraints for scale-up:** respect VRAM budgets on cloud GPUs (e.g. GCP L4/T4), use dynamic memory batching in the CUDA spatial-hash kernels so large multimeric complexes don't overflow VRAM, and keep any unavoidable CPU-bound steps strictly isolated from the GPU hot path.

---

## ⚠️ Limitations & Path to a Production-Ready Model

The infrastructure and training data are now real; the main open problems are scientific rather than mechanical:

- **Pocket ranking is unsolved.** Localization is excellent (99.8 % fragment coverage across all pockets), but ranking the single best pocket is weak (near-zero Pearson correlation of score vs. fragment count). Improving the ranking/regression target is the top priority.
- **Static "lock-and-key" snapshots.** Training on single rigid crystal structures ignores induced-fit and cryptic pockets. **Fix:** ingest structural ensembles or MD frames.
- **Solvent/co-factor blindness.** Crystallographic waters and some metal ions are stripped during sanitization, altering true local electrostatics.
- **Scalar features miss directionality.** Distance-weighted densities cannot capture H-bond geometry; an E(3)-equivariant network (above) is the planned successor.
- **Toward robustness:** broaden curated training data (scPDB, PDBbind, fragment screens) and add dynamic CUDA memory batching for very large complexes.

---

## 🧬 Project Evolution & Version History

This pipeline evolved from a localized, single-target proof-of-concept into a proteome-scale, leakage-audited zero-shot prediction engine.

### Version 1 — The Prototype (`xgboost_gpu_(3)_(4)_(2).py`)
- **Features Added:** C++ CUDA spatial hashing, RDKit data parsing, HDBSCAN spatial clustering.
- **Technical Explanation:** A zero-copy pipeline mapped RDKit-parsed PDB topologies into a 3D bounding box; custom CUDA kernels (`calc_hash`, `find_cell_bounds`) computed local pharmacophore densities, followed by cuML HDBSCAN for pocket segmentation.
- **Non-Technical Summary:** We treated the protein like a 3D grid and let the GPU rapidly scan its surface, compute chemical properties, and group promising spots into pockets.
- **Outcome:** Worked well for its single training protein but failed to generalize — it effectively memorized one shape.

### Version 2 — The Reality Check (`mpro_copy_of_xgboost_gpu.py`)
- **Features Added:** Zero-shot inference on an unseen target (SARS-CoV-2 MPro), fragment-intersection logic, Pearson correlation scoring.
- **Technical Explanation:** Added a validation pipeline using `scipy.spatial.cKDTree` to map predicted centroids against real XRD fragment coordinates, graded with `scipy.stats.pearsonr`.
- **Outcome:** Model failed (negative Pearson) due to domain shift — having seen only one protein, it hadn't learned universal chemistry. This proved the need for a large, diverse training set.

### Version 3 — The Pivot to Big Data (`scpdb_data_copy_of_xgboost_gpu.py`)
- **Features Added:** Extraction/integration of the scPDB database (16,000+ structures), multi-target batch scripts.
- **Outcome:** Foundation for generalized learning, but exposed severe CPU/memory bottlenecks when using RDKit to load thousands of proteins sequentially.

### Version 4 — The Engineering Breakthrough (`scpdb_data_version_3_xgboost_gpu.py`)
- **Features Added:** Native `.mol2` text parser, `process_folder()` batch loop with aggressive VRAM flushing, dynamic `scale_pos_weight`.
- **Technical Explanation:** Removed RDKit from the hot path in favor of a native text parser feeding CuPy directly; corrected the ~1:64 class imbalance via `scale_pos_weight`; used `free_all_blocks()` to prevent VRAM overflow.
- **Outcome:** Processed millions of surface points without crashing; the penalty weight sharply improved detection of true binding sites on unseen targets.

### Version 5 — The Enterprise Upgrade
- **Features Added:** Out-of-core incremental XGBoost training (`.npy` chunks on disk), strict UniProt-based holdout to prevent data leakage.
- **Technical Explanation:** An out-of-core loop handled the ~39 GB matrix payload; XGBoost was warm-started chunk-by-chunk via `xgb_model`. Leakage was eliminated by querying RCSB for all PDB IDs sharing a target's UniProt ID and banning them from training.
- **Outcome:** A scientifically valid, leak-audited universal model producing fast zero-shot predictions.

### Version 6 — Robustness & Scale Hardening (`scPDB_Data_version_6_XGBoost_GPU.ipynb`)
- **Features Added:** Larger/steadier batch ingestion and VRAM-management refinements consolidating the out-of-core chunked training loop and holdout firewall introduced in Version 5.
- **Outcome:** A more stable proteome-scale training run, setting up the fully audited Version 7 pipeline.

### Version 7 — The Audited Universal Model (`scPDB_Data_version_7_XGBoost_GPU.ipynb`) — **current**
- **Features Added:** Real ground-truth labelling from co-crystallized `ligand.mol2` via `cKDTree`; a two-layer data-leakage firewall (strict 101-ID holdout **including `7t2t`** + UniProt homology audit); incremental out-of-core GPU training with per-chunk `scale_pos_weight`; zero-shot evaluation on apo MPro (`7T2T`); and a large-scale inference test over the **422-protein holdout set** with per-target PR-AUC.
- **Technical Explanation:** Every SAS point is labelled by real experimental binding (`≤ 2.5 Å` from a ligand atom). Training streams `X_chunk_*.npy`/`Y_chunk_*.npy` and warm-starts one booster across chunks. Before ingestion, holdout base IDs are skipped by folder name and cross-checked against RCSB UniProt homologs, and the audit confirms a strictly zero training/test intersection.
- **Outcome:** A universal, leakage-audited pocket predictor whose weights (`data_leak_audited_universal_scPDB_pocket_weights.pkl`) localize essentially all fragment sites (99.8 % coverage across pockets) on strictly unseen targets in seconds — with pocket *ranking* identified as the key remaining challenge.

---

## 🧰 Environment & Dependencies

- **Runtime:** an NVIDIA-GPU-enabled environment (developed on **Google Colab** GPU runtimes; paths such as `/content/...` reflect this).
- **GPU stack:** CuPy, cuML (RAPIDS, provides GPU HDBSCAN), XGBoost with GPU support.
- **Installed by the notebook:** `rdkit`, `py3Dmol`, `pdb2pqr`, `alphashape`, `trimesh`, `MDAnalysis`.
- **Also used:** NumPy, SciPy (`cKDTree`), scikit-learn (`average_precision_score`), pandas, seaborn/matplotlib, pickle, json, `urllib` (RCSB API).
- **Data:** the scPDB database (`scPDB.tar.gz`), extracted under `/content/pipeline_working_dir/scPDB/scPDB` with one subfolder per complex.
- `update_notebook.py` is a small helper for programmatically editing notebook cells.

---

## ▶️ How to Run

1. Open **`scPDB_Data_version_7_XGBoost_GPU.ipynb`** in a **GPU-enabled Colab (or Jupyter) runtime** with the RAPIDS/CuPy/XGBoost-GPU stack available.
2. **Phase 0–2 (train):** provide/extract `scPDB.tar.gz`, run the holdout + leakage-audit cells, then the batch pipeline and incremental training cells to produce `data_leak_audited_universal_scPDB_pocket_weights.pkl`. *(To skip retraining, load the provided pre-trained weights archive instead.)*
3. **Phase 3 (infer):** the default target is the apo SARS-CoV-2 main protease `7T2T`, downloaded automatically. Change the PDB ID to analyze a different protein.
4. **Phase 4–5 (evaluate):** run the analytics, fragment-coverage, Pearson, leakage re-audit, and 422-protein large-scale test cells.
5. Inspect the printed diagnostics and inline `py3Dmol` visualizations, and collect the exported JSON artifacts (below).

---

## 📦 Output Files

| File | Produced by | Contents |
|------|-------------|----------|
| `holdout_test_set.json` | Holdout / audit cells | Reserved holdout base IDs (incl. `7t2t`) + any homologs flagged by the UniProt audit |
| `X_chunk_*.npy` / `Y_chunk_*.npy` | Batch pipeline | Out-of-core feature/label chunks for incremental training |
| `data_leak_audited_universal_scPDB_pocket_weights.pkl` (`.tar.gz`) | Training | Pickled universal XGBoost model trained on scPDB |
| `7T2T.pqr` / prepared PDB | PDB2PQR | Protonated inference target with AMBER charges |
| `7T2T_gpu_predictions.json` | Inference | Ranked pockets (P2Rank-style: rank, score, centroid, volume) |
| `7T2T_pocket_residues.json` | Inference | Binding-site residues for the top pockets |
| `pocket1_pharmacophore_features.json` | Inference | 3-D pharmacophore map of the top pocket |
| `MPro alligned XRD Fragment Coordinates.json` | Evaluation | Real fragment coordinates used for coverage/Pearson analysis |
| `inference_results.json` / `holdout_test_set_inference_results/` | Large-scale test | Per-target predictions and PR-AUC over the 422-protein holdout set |

---

*This project is a research pipeline for proteome-scale, GPU-accelerated druggable-pocket discovery, trained on real scPDB ground truth and audited against data leakage, intended to feed downstream virtual-screening and generative drug-design campaigns.*
