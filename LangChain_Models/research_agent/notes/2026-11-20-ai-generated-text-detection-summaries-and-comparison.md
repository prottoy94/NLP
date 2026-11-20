## Summaries

### 1. ‘Your AI Text is not Mine’: Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions (arXiv:2606.04906v2)
- **Problem:** Lack of unified definition for AI‑generated text leads to misaligned datasets and detectors.
- **Method:** Proposes a taxonomy of seven AI‑generated text detection (AITD) notions, builds the AITDNA benchmark (350 texts, 99 authors, multi‑genre, multiple LLMs/temperatures), and evaluates existing detectors across these notions.
- **Datasets:** AITDNA.
- **Key Results:** Detectors excel only for specific notions; prior benchmarks miss realistic human‑AI co‑creation. Notion‑aware evaluation reveals hidden assumptions affecting performance.
- **Limitations:** Not fully detailed in summary (insufficient info).
- **Key Equations:** \(gen = (au, pr)\) – genesis labeling; \(ar(x)\) – AI token ratio; \(\tau\) – AI ratio threshold.

### 2. PASA: A Principled Embedding‑Space Watermarking Approach for LLM‑Generated Text under Semantic‑Invariant Attacks (arXiv:2605.10977v2)
- **Problem:** Vocabulary‑space watermarks break under semantic‑invariant attacks (e.g., paraphrasing).
- **Method:** Embeds watermark in semantic embedding space using token clustering and synchronized randomness keyed to recent semantic history; two‑stage sampling ensures low false‑positive rates while preserving generation distribution.
- **Datasets:** Not specified in the paper.
- **Key Results:** PASA outperforms vocab‑space baselines in AUC‑ROC and TPR@1%FPR across paraphrase attacks, maintaining fluency.
- **Limitations:** Not fully detailed (insufficient info).
- **Key Equations:** \(Q_{X_t|x_{t-1},p_t}\) – next‑token prediction distribution; joint distribution \(Q_{X^T}=\prod_{t=1}^{T} Q_{X_t|X_{t-1}}\); false‑alarm threshold \(\alpha=0.4\).

## Comparison Table (Focus: Detection Method & Robustness)

| Paper | Detection Method | Datasets | Robustness to Attacks | Key Result | Limitations |
|---|---|---|---|---|---|
| **‘Your AI Text is not Mine’** (arXiv:2606.04906v2) | Taxonomy‑based benchmark evaluation of existing detectors across seven AITD notions. | AITDNA (350 texts, 99 authors, multi‑genre, multi‑model, temperature variants). | Evaluates detectors under varied genesis assumptions; does **not** propose a new robust detector. | Existing detectors are narrowly tuned; performance drops when notions change. | Limited detail on some experimental settings. |
| **PASA** (arXiv:2605.10977v2) | Embedding‑space watermarking via semantic clusters and synchronized randomness. | Not specified. | Designed to survive semantic‑invariant attacks (paraphrasing, T5‑based replacement, DIPPER levels). | Higher AUC‑ROC and TPR@1%FPR than vocab‑space baselines while preserving naturalness. | Dataset details missing; broader evaluation not provided. |

**Overall Insight**
- The first paper highlights the *evaluation* gap: detectors are often assessed on unrealistic assumptions, leading to over‑optimistic performance reports. It calls for notion‑aware benchmarks like AITDNA.
- The second paper addresses the *method* gap by introducing a watermark that remains detectable after strong meaning‑preserving transformations, demonstrating concrete robustness improvements.
- Together, they suggest a research direction: develop detection methods (e.g., watermarking) that are evaluated on notion‑aware, realistic benchmarks to ensure both effectiveness and robustness in real‑world co‑creation scenarios.
