# Meta-Learning for Few-Shot Histopathology Classification

Notebooks for the Master's thesis *Meta-Learning for Rapid Adaptation and Generalization in Medical Image Classification* (Deradra Rayenne Maissoune, Ferhat Abbas University Sétif 1, 2025–2026).

The thesis compares Prototypical Networks, MAML and MoCo v2 self-supervised pre-training for few-shot classification of histopathology patches, in-domain and when transferring from colorectal to prostate tissue.

## Results

All experiments use a ResNet-18 (ImageNet weights, `conv1`–`layer3` frozen), 128×128 inputs, 5 support and 10 query images per class, and 100 test episodes.

| # | Notebook | Method | Trained on | Tested on | Test accuracy |
|---|---|---|---|---|---|
| 1 | `protonet-crc.ipynb` | ProtoNet | CRC | CRC, 5-way | 97.24% |
| 2 | `prontonet-sicapv2.ipynb` | ProtoNet | SICAPv2 | SICAPv2, 2-way | 95.45% |
| 3 | `ssl-cross-domain-protonet-cross-domain.ipynb` (second half) | MoCo v2 encoder + ProtoNet head, two stages | CRC, then SICAPv2 | SICAPv2, 2-way | 89.35% |
| 4 | `ssl-cross-domain-protonet-cross-domain.ipynb` (first half) | MoCo v2 encoder + ProtoNet head | CRC (unlabelled), then SICAPv2 | SICAPv2, 2-way | 95.05% |
| 5 | `maml-crc.ipynb` | MAML | CRC | CRC, 5-way | 87.28% |
| 6 | `maml-scapv2.ipynb` | MAML | SICAPv2 | SICAPv2, 2-way | 73.70% |
| 7 | `maml-crossdomain.ipynb` | MAML | CRC | SICAPv2, 2-way | 65.55% (65.7 ± 2.2% over 5 seeds) |

Random baselines are 20% for 5-way and 50% for 2-way tasks.

## What each experiment does

**Experiments 1 and 2 — ProtoNet, in-domain.** `layer4` of the ResNet-18 is trained episodically. Each class prototype is the mean of its support embeddings, and queries are assigned to the nearest prototype by squared Euclidean distance. Adam, learning rate 1e-3, StepLR (step 10, γ 0.5), 20 epochs of 400 episodes; Experiment 2 adds early stopping with patience 6.

**Experiment 4 — MoCo v2 pre-training, then ProtoNet on SICAPv2.** The encoder is pre-trained with MoCo v2 on 20,000 unlabelled CRC patches (20 epochs, queue 4096, temperature 0.07, momentum 0.999) and then frozen. A small projection head is trained on labelled SICAPv2 episodes (up to 30 epochs, patience 8). The encoder never sees prostate tissue or any labels; the projection head does.

**Experiment 3 — two-stage transfer with the same MoCo encoder.** Stage 1 trains the projection head on 5-way CRC episodes only; evaluated directly on SICAPv2 this gives 59.20%. Stage 2 fine-tunes the head on labelled SICAPv2 episodes for 20 epochs at a ten times lower learning rate, giving 89.35%.

**Experiments 5 and 6 — MAML, in-domain.** `layer4` and a linear head are meta-learned with the [`higher`](https://github.com/facebookresearch/higher) library: 5 inner SGD steps at learning rate 0.01, Adam outer optimiser with cosine annealing, meta-batch of 2 episodes. Gradients are backpropagated through the inner loop. Experiment 6 uses an outer learning rate of 5e-5, dropout 0.3 on the head, weight decay 1e-4 and early stopping with patience 6.

**Experiment 7 — MAML, cross-domain.** The model is meta-trained on 5-way CRC episodes only. At test time the head is replaced by a fresh 2-way head and adapted for 10 inner steps on each SICAPv2 support set. No prostate data is used in training. The notebook reports a single seeded run and a five-seed mean ± standard deviation.

## Data

- **NCT-CRC-HE-100K** (Kather et al., 2018): 100,000 H&E patches of colorectal tissue in 9 classes. All 9 classes appear in every split; images are split 70 / 15 / 15 at image level.
- **SICAPv2** (Silva-Rodríguez et al., 2020): prostate biopsy patches with pixel-level masks. A patch is labelled *cancer* if more than 30% of its mask is tumour, and *benign* if more than 50% is benign with under 10% tumour; other patches are discarded. Benign patches are capped at 2,000, giving 2,000 benign and 1,293 cancer patches, split 70 / 15 / 15 at patch level.

Because SICAPv2 is split by patch and not by slide or patient, patches from the same slide can appear in both training and test sets. The in-domain SICAPv2 numbers should be read with that in mind.

## Running

The notebooks were written for Kaggle with two Tesla T4 GPUs and PyTorch 2.10. Dataset paths at the top of each notebook point to Kaggle inputs and need changing elsewhere.

```
pip install torch torchvision higher numpy pillow scikit-learn matplotlib seaborn
```

Each notebook runs top to bottom and takes roughly one to two hours on a T4; the SSL notebook takes about four, and the five-seed section of `maml-crossdomain.ipynb` about five times a single run.

## Reproducibility

- Only `maml-crossdomain.ipynb` seeds PyTorch and the data loaders. The other notebooks fix the data split (seed 42) but sample episodes randomly, so a rerun gives accuracies close to the table, not identical.
- The saved outputs are from the original Kaggle runs. Some code cells were restored after those runs; rerun a notebook to regenerate its outputs from the code as published.
- Experiment 7 is a later rerun. The thesis reports 62.80% for its original run; the run saved here gives 65.55%.
- Accuracy is the mean over 100 test episodes of 10 query images per class.

## References

- Snell, Swersky & Zemel (2017). *Prototypical Networks for Few-shot Learning.* NeurIPS.
- Finn, Abbeel & Levine (2017). *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.* ICML.
- Chen, Fan, Girshick & He (2020). *Improved Baselines with Momentum Contrastive Learning.* arXiv:2003.04297.
- Kather et al. (2018). *100,000 histological images of human colorectal cancer and healthy tissue.* Zenodo.
- Silva-Rodríguez et al. (2020). *Going Deeper through the Gleason Scoring Scale.* Computer Methods and Programs in Biomedicine.
