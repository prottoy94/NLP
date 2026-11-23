# AI-Generated Text Detection: Literature Review

## Papers Reviewed

1. **2606.04906v2** - "Your AI Text is not Mine": Redefining and Evaluating AI-generated Text Detection under Realistic Assumptions (Dycke et al., 2026)
2. **2605.10977v2** - PASA: A Principled Embedding-Space Watermarking Approach for LLM-Generated Text under Semantic-Invariant Attacks (Ai & He, 2026)

---

## Summary: 2606.04906v2

**Problem:** The paper addresses a critical gap in AI-generated text detection (AITD) research: the lack of a systematic framework for defining detection tasks (notions) and the misalignment of existing benchmarks with real-world human-AI co-creation. Current datasets encode hidden assumptions about text genesis, attacker models, and normative standards, making cross-dataset comparison invalid and evaluation unreliable for policy-relevant scenarios.

**Method:** The authors make three contributions: (C#1) A formal framework structuring AITD notions along three dimensions—normative standard, genesis assumptions, and attacker model—identifying five existing and two novel notions (content-based and authorship-ID-based). (C#2) AITDNA, a notion-agnostic dataset of 362 texts from 99 authors collected via a dedicated co-writing interface (CARE) that logs full interaction traces (prompts, edits, accept/reject decisions) across multiple LLMs, temperatures, and writing scenarios (argumentative, creative, explanatory, scientific). (C#3) Controlled experiments on AITDNA and five prior datasets evaluating ten detectors (zero-shot, fine-tuned, proprietary) under aligned notion parameters.

**Datasets:**
- AITDNA (novel, 362 texts, 99 authors, detailed genesis logs)
- CoAuthor (Lee et al., 2022)
- SenDetEx (Jiang et al., 2025)
- BD (Zeng et al., 2024b)
- DetectRL (Wu et al., 2024)
- Mixset (Zhang et al., 2024)

**Key Results:**
- AITDNA exhibits natural co-creation patterns: ~51% AI tokens, 23.3 boundaries/text, high linguistic complexity; synthetic datasets differ substantially (e.g., DetectRL/Mixset: ~64% AI tokens, en-bloc genesis assumptions).
- Document-level AITD yields highest detection performance; performance drops with granularity (boundary > sentence > intent/content). Intent- and content-based AITD are most difficult for surface-feature detectors.
- No detector consistently outperforms across datasets when τ=0.5 is fixed; fine-tuned detectors (TMR, moBERT) excel on synthetic datasets but may benefit from data leakage; zero-shot methods (Log Rank, Likelihood) lead on AITDNA/CoAuthor.
- Threshold τ critically affects evaluation: F1 and FPR decrease monotonically with τ; optimal τ depends on target domain's AI-token distribution and policy requirements.
- Subsampling CoAuthor to AITDNA size shows detector rankings are stable (bias < 0.06 AUROC), supporting generalizability of AITDNA-scale findings.

**Limitations:**
- AITDNA collected under lab conditions, missing conversational LLM interaction paradigms (e.g., ChatGPT).
- Authorship-ID-based AITD could not be evaluated due to insufficient human texts per author.
- Proprietary detectors (GPTZero, Pangram) not tested on all datasets due to cost.
- Content/intent policies implemented via GPT-5.4-nano classifier; human validation limited to 70 sentences.

---

## Summary: 2605.10977v2

**Problem:** Existing LLM watermarking methods operate on token vocabulary and are vulnerable to semantic-invariant attacks (e.g., paraphrasing, synonym substitution) that preserve meaning but alter surface tokens, destroying token-level watermark signals. Heuristic semantic-aware methods improve robustness but sacrifice text quality (distortion) and lack principled guarantees on the trade-offs among detection accuracy, robustness, and distortion.

**Method:** PASA (Principled watermarking Approach under Semantic-invariant Attacks) operates in a latent semantic embedding space. It partitions the token embedding space into K disjoint semantic clusters via a mapping function f: V → [K] (using K-means on gte-Qwen2-7B-instruct embeddings). Watermark embedding uses a two-stage sampling strategy grounded in a theoretical framework characterizing the jointly optimal embedding-detection pair under a semantic-invariant attack model. Detection uses a surrogate language model (SLM) to approximate the NTP distribution, reconstructs the auxiliary distribution, and accumulates scores based on alignment between observed semantic clusters and re-sampled auxiliary variables synchronized via a secret key and semantic history window of size w.

**Datasets:**
- C4 (realnewslike subset)
- ELI5 (long-form QA)

**Key Results:**
- On clean text, PASA achieves near-perfect detection (AUROC ≈ 1.0, TPR@1%FPR ≈ 1.0) on Llama-2-13B and Mixtral-8×7B.
- Under T5-Large token replacement (Llama-2-13B): PASA AUROC=0.9946, TPR@1%FPR=0.9296 vs KGW (0.9750, 0.7350) and DAWA (0.8067, 0.3300).
- Under T5-XXL (Llama-2-13B): PASA AUROC=0.9392, TPR@1%FPR=0.8040 vs KGW (0.9196, 0.6800).
- Under DIPPER paraphrasing (Lex=60, Ord=80): PASA AUROC=0.8956, TPR@1%FPR=0.5377 vs KGW (0.8346, 0.3100), Exp-edit (0.5029, 0.1150), AWTI (0.7391, 0.1350), DAWA (0.3697, 0.0200).
- Text quality (PPL on GPT-NeoX-20B): PASA=11.44, Unwatermarked=12.41, Human=10.41, KGW=11.81, Exp-edit=23.40, AWTI=19.77, DAWA=8.41.
- Generation latency: 13.35s (vs 12.93s unwatermarked); Detection latency: 0.27s (lowest among robust methods).
- Ablation: Optimal cluster granularity K=4; robustness degrades for K>500. Optimal window size w=3; larger w increases seed sensitivity to perturbations.

**Limitations:**
- Degrades under very strong rewriting that changes semantic content and distributional structure.
- Detection on very short texts remains challenging due to insufficient statistical evidence.
- Requires tokenizer-compatible SLM for detection; tokenizer mismatch reduces cross-family transferability.

---

## Comparison Table

| Paper | Problem Focus | Detection Method | Robustness Evaluation | Datasets Used | Key Results | Limitations |
|---|---|---|---|---|---|---|
| [Your AI Text is not Mine](https://arxiv.org/abs/2606.04906v2) | Evaluates AI-text detection frameworks under realistic co-creation assumptions | Compares 10 detectors across 5 detection notions (doc, sentence, boundary, content, intent) | Tests across datasets with varying genesis; performance drops with granularity; τ threshold critical | AITDNA (362 texts, 99 authors, co-writing logs) + 5 prior datasets | Doc-level best; fine-tuned wins synthetic; zero-shot wins realistic; τ changes rankings | Lab conditions; no conversational LLM; authorship-ID unevaluated; proprietary detectors limited; GPT classifier for policy |
| [PASA](https://arxiv.org/abs/2605.10977v2) | Proposes robust semantic embedding-space watermarking against paraphrasing/token-replacement attacks | Embedding-space watermarking via semantic clusters; detection via surrogate LM and auxiliary distribution alignment | Robust to T5 replacement, DIPPER paraphrasing; degrades under strong semantic rewriting; short texts hard | C4 (realnewslike), ELI5 (long-form QA) | Near-perfect clean detection; SOTA robustness vs KGW/DAWA/AWTI; low distortion (PPL 11.44); fast detection (0.27s) | Degrades under strong semantic rewriting; short texts challenging; tokenizer-compatible SLM needed; limited cross-family transfer |

**Overall insight:** Paper 1 establishes evaluation framework for AI-text detection under realistic co-creation; Paper 2 proposes a specific watermarking method for robust detection. Different goals: benchmarking vs. method proposal.

---

## Key Takeaways

1. **Evaluation Framework vs. Detection Method**: The two papers address complementary aspects—2606.04906v2 provides a rigorous evaluation framework and realistic dataset (AITDNA), while 2605.10977v2 proposes a novel watermarking technique (PASA) for robust detection.

2. **Realistic Co-creation Matters**: 2606.04906v2 demonstrates that existing benchmarks encode unrealistic assumptions (en-bloc AI generation), while real co-creation involves interleaved human-AI writing with ~51% AI tokens and frequent boundaries.

3. **Semantic-Level Robustness**: 2605.10977v2 shows that operating in semantic embedding space (rather than token space) provides principled robustness against paraphrasing attacks while maintaining text quality.

4. **Threshold Sensitivity**: Both papers highlight the importance of detection thresholds—2606.04906v2 shows τ critically affects rankings, while 2605.10977v2's theoretical framework characterizes the fundamental trade-off between missed detection, false alarm, and distortion.

5. **Practical Deployment Considerations**: PASA achieves low detection latency (0.27s) and minimal generation overhead, while AITDNA reveals that zero-shot detectors generalize better to realistic co-creation scenarios than fine-tuned ones.