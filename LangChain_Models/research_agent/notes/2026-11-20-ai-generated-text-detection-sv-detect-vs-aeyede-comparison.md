# AI-Generated Text Detection: SV-Detect vs AEyeDE

## Paper 1: SV-Detect: AI-generated Text Detection with Steering Vectors
**arXiv:** [2606.07313v2](https://arxiv.org/abs/2606.07313v2) · **Published:** 2026-06-05 · **Authors:** Mikhail Vishnyakov, Tatiana Gaintseva

### Problem
Detecting AI-generated text under distribution shift (cross-domain, cross-source model, editing attacks) remains challenging. Existing detectors relying on token-level statistics, likelihood scores, or supervised classifiers on text often degrade under transfer, suggesting reliance on generator-specific artifacts.

### Method
SV-Detect extracts layer-wise steering vectors from a frozen LLM's hidden representations to separate human-written from AI-generated text. For each layer \(l\), a steering vector \(\mathbf{v}_l\) is constructed from training activations. A text \(x\) is represented by its cosine similarity projections \(s_l(x)\) onto these vectors across all layers, forming a compact feature vector \(\mathbf{s}(x) = [s_1(x), \dots, s_L(x)]\). A lightweight logistic regression classifier on standardized \(\mathbf{s}(x)\) yields the detection probability.

### Datasets
- DetectRL (Wu et al., 2024): Multi-Domain (ArXiv, XSum, Writing, Review), Multi-LLM (GPT-3.5, Claude, PaLM-2, Llama-2), Multi-Attack (prompt, paraphrase, perturbation, mixing).
- MIRAGE (Fu et al., 2025): DIG (Disjoint-Input Generation) and SIG (Shared-Input Generation) settings with GENERATE, POLISH, REWRITE transformations.
- PADBen (Zha et al., 2025): Zero-shot cross-benchmark target for paraphrase laundering and human-edited AI text.
- COLING-2025 MGT English split (Wang et al., 2025): Large-scale multilingual benchmark with 27 generators.
- RAID (Dugan et al., 2024): Used for matched-backbone comparison with RepreGuard.

### Key Results
- In-distribution: Near-perfect AUROC (99.8–100.0) and F1 (98.8–100.0) on DetectRL; best on MIRAGE GENERATE (AUROC 0.9777 DIG, 0.9779 SIG), competitive on POLISH/REWRITE.
- Cross-source generalization (DetectRL): SV-Detect maintains ≥92 F1 on Multi-LLM and ≥95 F1 on Multi-Attack, outperforming RoBERTa-Base (7–14 F1 points gain on Multi-Domain) and zero-shot baselines.
- Cross-benchmark transfer: SV-Detect is best or second-best in all four transfer directions (MIRAGE→DetectRL, DetectRL→MIRAGE, MIRAGE→PADBen, DetectRL→PADBen), substantially outperforming zero-shot detectors and matching RoBERTa-Large.
- Ablations: Logistic-regression steering vectors dominate mean-difference and PCA (strongest average/worst-case transfer). Backbone choice affects robustness (Qwen3-1.7B-Base most robust), but method generalizes across LMs. Logistic regression classifier is most reliable under transfer vs. CatBoost/KNN.
- Interpretability: Steering directions align with lexical/stylistic cues (formal/technical tokens on AI side, casual/punctuation on human side) but capture 13–24 AUROC points beyond regex-based stylistic features. Cross-LM logit-lens shows consensus on LLM-polished register (e.g., 'endeavors', 'utilization') across GPT-Neo, Qwen3, Gemma.
- COLING-2025: Llama-2-7B backbone with LogReg steering and trimmed generator pool achieves top leaderboard Macro-F1 84.8, Accuracy 85.00.

### Limitations
- Depends on probe LLM; representation geometry affects performance (backbone sensitivity remains).
- Requires full LLM forward pass; less efficient than small encoder-based detectors (RoBERTa ~7–11 ms vs. SV-Detect ~25.7 ms on A100).
- Evaluated on English benchmarks; does not cover multilingual or mixed human–AI authorship (localized AI spans, iterative collaboration).
- Produces document-level score; no span-level attribution.
- Supervised; requires labeled data and periodic recalibration as generators/styles evolve.

### Key Equations
- Mean-pooled layer representation: \(\mathbf{a}_l(x) = \frac{1}{T} \sum_{t=1}^T \mathbf{H}_l(x)_t \in \mathbb{R}^d\).
- Mean-difference steering vector: \(\mathbf{v}_l^{\text{mean}} = \frac{\boldsymbol{\mu}_l^{(f)} - \boldsymbol{\mu}_l^{(r)}}{\|\boldsymbol{\mu}_l^{(f)} - \boldsymbol{\mu}_l^{(r)}\|_2}\), where \(\boldsymbol{\mu}\) are class means.
- Logistic-regression steering vector: \(\mathbf{v}_l^{\text{logreg}} = \frac{\mathbf{w}_l}{\|\mathbf{w}_l\|_2}\), \(\mathbf{w}_l\) from per-layer logistic regression on activations.
- PCA steering vector: \(\mathbf{v}_l^{\text{PCA}} = \text{PC1}(\{\mathbf{a}_l(x^{(f)}_j) - \mathbf{a}_l(x^{(r)}_i)\})\), leading principal component of paired differences.
- Projection feature (cosine similarity): \(s_l(x) = \frac{\langle \mathbf{a}_l(x), \mathbf{v}_l \rangle}{\|\mathbf{a}_l(x)\|_2}\).
- Detection head: \(p(y=1|x) = \sigma(\mathbf{w}^\top \tilde{\mathbf{s}}(x) + b)\), \(\tilde{\mathbf{s}}\) standardized, \(\sigma\) sigmoid.
- Logit-lens decoding: \(\text{logits}_l = \mathbf{W}_U^\top \text{LN}_f(\mathbf{v}_l)\), projecting steering vector through final layer norm and unembedding.

### Takeaway
SV-Detect frames AI-text detection as a representation-space probing problem: layer-wise steering vectors from a frozen LLM yield compact, interpretable features that achieve strong in-distribution and out-of-distribution generalization across domains, source models, and editing attacks, capturing both stylistic cues and deeper representation-level signals.

---

## Paper 2: AEyeDE: An Attention-Based Attribution Framework for AI-Generated Text Detection
**arXiv:** [2606.00016v1](https://arxiv.org/abs/2606.00016v1) · **Published:** 2026-04-13 · **Authors:** Aria Nourbakhsh, Adelaide Danilov, Christoph Schommer, Salima Lamsiyah

### Problem
Detecting AI-generated text is increasingly difficult as modern LLMs approach human-level fluency and evade detectors relying on surface statistics (perplexity, burstiness) or likelihood-based signals. Existing methods—statistical, likelihood-based, supervised classifiers, and watermarking—suffer from degradation under RLHF optimization, domain shift, unseen generators, and fragility to paraphrasing/post-editing. The paper hypothesizes that token-token interaction structure captured by attention-based attribution maps provides a richer, more robust detection signal than 1D token-level statistics.

### Method
AEyeDE extracts attention-derived attribution matrices from a proxy Transformer model G_θ with white-box access and processes them with a lightweight CNN classifier.

**Attribution Map Extraction**:
For an L-layer, H-head Transformer, query/key matrices Q^(ℓ,h) ∈ R^{T_t×d_k}, K^(ℓ,h) ∈ R^{T_s×d_k} yield per-head attention:
\[
\tilde{A}^{(ℓ,h)}(x;θ) = \text{softmax}\left(\frac{Q^{(ℓ,h)}(K^{(ℓ,h)})^⊤}{\sqrt{d_k}} + C\right) \in \mathbb{R}^{T_t \times T_s}
\]
where C is a causal mask for decoder-only models (none for encoder-decoder cross-attention). The final attribution map averages over all layers and heads:
\[
A(x;θ) = \frac{1}{LH} \sum_{ℓ=1}^L \sum_{h=1}^H \tilde{A}^{(ℓ,h)}(x;θ) \in \mathbb{R}^{T_t \times T_s}
\]
Each entry a_{t,s} gives average attention mass from target position t to source/previous position s.

**Input Processing**:
- Encoder-decoder: full cross-attention map (≤128×128 tokens).
- Decoder-only: sample fixed-size square blocks (w_a=128) along the main diagonal where attribution mass concentrates.
Validity mask P = m_t m_s^⊤ marks non-padding entries; a block is valid if it contains ≥1 non-padding entry.

**Attribution Encoder E_attr**:
Treats each block A_k ∈ R^{w_a×w_a} as a single-channel image. CNN channel progression: 1→32→64→128→128→256 with 2×2 max-pooling, then global average pooling and a 2-layer MLP:
\[
r_k = g(A_k) \in \mathbb{R}^{256}, \quad b_k = W_2 \rho(\text{BN}(W_1 r_k + b_1)) + b_2
\]
Final convolutional feature map is 16×16; each spatial cell corresponds to an 8×8 patch of the original block (used for motif analysis).

**Optional Text Branch**:
Text split into chunks of size w_t, encoded by E_text to vectors c_i. Both branches aggregated via learnable attention pooling:
\[
s_i = ω^⊤ \tanh(W_p z_i), \quad α_i = \frac{m_i \exp(s_i)}{\sum_j m_j \exp(s_j)}, \quad \text{Pool}(\{z_i\},\{m_i\}) = \sum_i α_i z_i
\]
Final representation h = [h_text; h_attr] (or h_attr alone) fed to a 2-layer MLP for binary classification:
\[
ℓ = w^⊤ \rho(W_h h + b_h) + b, \quad p(y=1|x,A) = σ(ℓ)
\]
Trained with binary cross-entropy.

### Datasets
- Encoder-decoder: WMT14 (fr-en, de-en) and UN Parallel Corpus (ar-en); 200k source-target pairs per language pair, gold references vs. Marian-MT translations, max 128 tokens.
- Decoder-only: HC3 (ChatGPT vs. human, ~24k/class, ≤1024 tokens), RAID (multi-generator: Cohere, Llama 2 70B, GPT-2 XL, Mistral-7B; downsampled to 26.7k per model, 12.9k human), Beemo (external test, 2.178 human + 2.178 machine from multiple LLMs).

### Key Results
- **Encoder-decoder translation detection** (Table 1): AEyeDE (CNN) outperforms text-only baseline on all language pairs: ar-en +3.6 F1 (74.6 vs 71.0), de-en +6.7 F1 (81.5 vs 74.8), fr-en +8.3 F1 (85.1 vs 76.8). Adding text (CNN+text) gives smaller further gains (+1.4–2.0 F1).
- **Generator-specific (individual) setting on RAID** (Table 2): AEyeDE achieves near-ceiling performance across all four generator families (F1 96.34–99.25, AUC 98.81–99.86), top on every generator. Large gains on GPT-neo (97.96 vs 83.70 RoBERTa) and Mistral (96.34 vs 82.71 RoBERTa). Maintains high precision and recall; baselines often show recall≈100% but precision≈class prior (~67%).
- **Cross-generator transfer (unified) setting** (Table 3): Trained on all RAID generators except Mistral, tested on Mistral. AEyeDE achieves best accuracy, AUC (78.86–80.67), and TPR@FPR=0.01 (56.80–58.97) for all generators. Operates conservatively: high precision (90.98–96.57), lower recall (61.06–65.55), F1 74.72–76.19. Baselines tend toward all-positive regime (recall≈100%, precision≈67%). RoBERTa strongest balanced baseline but trails on AUC and low-FPR metric.
- **Adversarial robustness** (Tables 5–8):
  - Paraphrasing (individual, GPT-neo): AEyeDE best F1 (81.50) and AUC (77.64), precision 74.78 vs 68.87 (RoBERTa).
  - Paraphrasing (unified): advantage diminishes; all methods near majority-class baseline, AEyeDE AUC 62.21 < RoBERTa 72.87.
  - Alternative-spelling (individual): AEyeDE best F1 (98.47), AUC (99.75), TPR@FPR=0.01 (96.26); next-best RoBERTa F1 84.06.
  - Alternative-spelling (unified): AEyeDE best accuracy (70.34), F1 (81.33), AUC (83.48), TPR@FPR=0.01 (59.10).
- **Cross-dataset generalization on Beemo** (Table 9): AEyeDE highest accuracy (66.36), precision (62.16), F1 (71.32), AUC (74.43). Baselines show near-chance accuracy/precision with recall>96% (poor calibration). GLTR matches AUC (74.38) and beats TPR@FPR=0.01 (15.46 vs 14.55).

### Limitations
- Requires white-box access to a proxy Transformer (though attribution patterns generalize across generator families, exact generator access not strictly required).
- Focuses on attention-based attributions; gradient-based saliency not explored due to computational overhead.
- Attention attribution computed by averaging over all layers and heads; fine-grained selection strategy left for future work.

### Key Equations
- Attribution map: A(x;θ) = (1/LH) Σ_ℓ Σ_h softmax(Q^(ℓ,h) K^(ℓ,h)^⊤ / √d_k + C)
- Block extraction: A_k = A[p_k:p_k+w_a, q_k:q_k+w_a]
- Validity mask: P = m_t m_s^⊤, block valid if Σ_{i,j} P[p_k+i, q_k+j] > 0
- CNN encoding: r_k = g(A_k), b_k = W_2 ρ(BN(W_1 r_k + b_1)) + b_2
- Attention pooling: s_i = ω^⊤ tanh(W_p z_i), α_i = m_i exp(s_i) / Σ_j m_j exp(s_j), Pool = Σ_i α_i z_i
- Classification: ℓ = w^⊤ ρ(W_h h + b_h) + b, p = σ(ℓ)
- Motif rate: r_s(c) = (1/|I_s|) Σ_{(k,u,v)∈I_s} 1[c_{k,u,v}=c], Δr(c) = r̄_machine(c) - r̄_human(c)

### Takeaway
AEyeDE leverages attention attribution maps from a proxy Transformer as structured 2D inputs to a CNN, achieving strong generator-specific detection, competitive cross-generator transfer, robustness to character-level spelling perturbations, and revealing interpretable local attention motifs that systematically differ between human and AI text.

---

## Comparison Table

| Paper | Detection Method | Robustness Focus | Datasets Used | Key Results | Limitations |
|---|---|---|---|---|---|
| [SV-Detect](https://arxiv.org/abs/2606.07313v2) | Steering vectors from frozen LLM hidden states; cosine similarity projections + logistic regression | Strong cross-domain, cross-source model, editing attacks; backbone sensitivity noted | DetectRL, MIRAGE, PADBen, COLING-2025, RAID | Near-perfect in-distribution; best cross-benchmark transfer; interpretable features | Probe LLM dependent; slower than encoders; English only; document-level; supervised |
| [AEyeDE](https://arxiv.org/abs/2606.00016v1) | Attention attribution maps from proxy Transformer; CNN classifier on 2D blocks | Robust to spelling perturbations; competitive cross-generator transfer; weak on paraphrase unified | WMT14, UN Parallel, HC3, RAID, Beemo | Near-ceiling generator-specific; strong cross-dataset; reveals attention motifs | Requires white-box proxy; averages all layers/heads; no gradient saliency |

**Overall insight:** SV-Detect uses representation-space probing for broad OOD generalization; AEyeDE exploits attention structure for generator-specific strength and interpretability. SV-Detect excels in transfer; AEyeDE in adversarial character-level robustness.