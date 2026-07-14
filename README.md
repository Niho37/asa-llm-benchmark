# Benchmarking LLMs for ASA Physical Status Classification from German Anesthesia Records

Code and prompts accompanying the paper:

> D. N. Bigler, A. Vogt, K. Riesen: *Benchmarking LLMs for ASA Physical Status Classification from German Anesthesia Records.* 12th IAPR TC3 Workshop on Artificial Neural Networks in Pattern Recognition (ANNPR 2026), Milan, Italy. Springer LNAI.

## Overview

We formulate ASA Physical Status classification (ordinal, 1–5) from real-world German-language clinical free text as a pattern recognition problem and benchmark six LLMs spanning five architectural paradigms under zero-shot and few-shot prompting:

| ID | Model | Category | Params | Fine-tuned |
|----|-------|----------|--------|------------|
| A | Phi-4 Reasoning Plus | Reasoning | 14B | – |
| B | Mistral Small 24B Instruct | General-purpose | 24B | – |
| C | Qwen2.5-14B Instruct | Multilingual | 14B | – |
| D | Llama-3.1-8B Instruct + QLoRA | Fine-tuned | 8B | ✓ |
| E | Qwen2.5-7B Instruct + QLoRA | Fine-tuned | 7B | ✓ |
| F | Meditron3-7B | Medical-domain | 7B | – |

**Key results (test set, n = 635, prompt P2):** Llama-3.1-8B + QLoRA achieves the best overall quadratic weighted kappa (W-κ = 0.594, zero-shot). QLoRA fine-tuning yields gains of up to ΔW-κ = +0.220 and enables compact models to outperform substantially larger pretrained models. A single clinically motivated prompt rule (temporality of historical diagnoses) outperforms all multi-rule combinations.

## Repository Structure

```
prompts/                        Full German-language prompt texts (P0–P5)
  P0_baseline.txt               ASA class definitions + output format only
  P1_asa5_restriction.txt       P0 + ASA-5 restriction rule
  P2_temporality_FROZEN.txt     P0 + temporality rule  ← selected & frozen
  P3_asa34_boundary.txt         P0 + ASA-3/4 decision rule
  P4_combined_P1_P2.txt         P0 + rules 1+2
  P5_combined_P1_P2_P3.txt      P0 + rules 1+2+3
scripts/
  01_prepare_inputs.py          Input text construction & chronological split
  02_inference.py               LLM inference (6 models, prompt registry, checkpointing)
  03_finetune.py                QLoRA fine-tuning (models D & E, Unsloth)
  04_evaluate.py                Metrics, confusion matrices, bootstrap CIs
```

## Prompt Ablation Design

Each prompt variant P1–P3 adds exactly **one** decision rule to the baseline P0 and is treated as an independent clinical hypothesis test (non-cumulative). P4/P5 test rule combinations. Prompt selection was performed exclusively on a stratified validation subset (val-dev, n = 100) using Model B; P2 was frozen before any test-set evaluation.

## Reproducibility

- Deterministic inference: temperature = 0, greedy decoding, repetition penalty 1.1, seed 42
- All models loaded in 4-bit NF4 quantization (single NVIDIA A40, 24 GB VRAM)
- QLoRA: rank 16, α = 32, dropout 0.05, 3 epochs, effective batch size 16, lr 2e-4, 8-bit AdamW, cosine schedule
- Pinned environment: `unsloth==2025.6.2`, `trl==0.8.6`, `transformers==4.52.4`
- Primary metric: quadratic weighted Cohen's kappa; 95% bootstrap confidence intervals

Replace `PATH/TO/DATA` and `PATH/TO/MODEL_CACHE` in the scripts with your local paths.

## Data Availability

The underlying clinical dataset (4,224 anonymized anesthesia cases from the Patient Data Management System of Inselspital Bern University Hospital, 2020–2024) **cannot be publicly shared**. The study was approved under the retrospective ethics protocol KEK Bern 2026-00331 (AISPM-Retro), which does not permit public release of patient-level data. Data may be made available upon reasonable request, subject to ethics approval and a data use agreement.

The few-shot examples used in the paper were drawn from the training split and are therefore not included in this repository. The prompt files contain the complete system prompts; few-shot examples follow the input schema documented in `scripts/01_prepare_inputs.py`.

## Citation

```bibtex
@inproceedings{bigler2026asa,
  author    = {Bigler, Daniel Nico and Vogt, Andreas and Riesen, Kaspar},
  title     = {Benchmarking {LLMs} for {ASA} Physical Status Classification from {German} Anesthesia Records},
  booktitle = {Artificial Neural Networks in Pattern Recognition (ANNPR 2026)},
  series    = {Lecture Notes in Artificial Intelligence},
  publisher = {Springer},
  year      = {2026}
}
```

## Acknowledgements

This work was supported by an intramural grant of the Department of Anaesthesiology, Inselspital Bern University Hospital.
