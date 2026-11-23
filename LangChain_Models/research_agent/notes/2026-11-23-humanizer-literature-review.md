## Summaries

### 1. Agents Can Use Base Models to Evade AI Detection (arXiv:2609.31876v1)
**Problem:** Current AI‑text detectors can reliably identify outputs from fine‑tuned or watermarked language models, and existing paraphrasing attacks cause semantic drift, limiting their usefulness for preserving task fidelity.
**Method:** Introduces HALO, a coding agent that samples continuations from a base language model and assembles final responses using `<copy>` tags that reference stored base LM snippets. Post‑processing resolves tags, yielding outputs composed up to 90 % of base LM tokens while preserving task relevance.
**Datasets:** Mythos, FACTS, HealthBench, IFEval.
**Key results:** Detection rate of commercial detector Pangram drops from 77 % to 24 %; task performance stays >90 % of baseline; query cost rises up to 30×.
**Limitations:** High computational/monetary cost; incomplete evasion on HealthBench; requires local base model.

### 2. PADBen: A Comprehensive Benchmark for Evaluating AI Text Detectors Against Paraphrase Attacks (arXiv:2511.00416v1)
**Problem:** Lack of systematic evaluation of detectors against realistic, multi‑iteration paraphrase attacks.
**Method:** Constructs PADBen benchmark with five text taxonomies (human, LLM‑generated, human‑paraphrased, LLM‑paraphrased, iteratively LLM‑paraphrased) and five detection tasks. Uses dual‑space representation analysis to expose an "intermediate laundering region" where paraphrased text evades classifiers.
**Datasets:** PADBen benchmark (350+ texts across genres).
**Key results:** State‑of‑the‑art detectors drop from >90 % accuracy on direct LLM output to near‑random on iteratively paraphrased text.
**Limitations:** Not fully detailed in the paper.

### 3. Adversarial Paraphrasing: A Universal Attack for Humanizing AI‑Generated Text (arXiv:2506.07001v2)
**Problem:** Need a training‑free, universal method to humanize AI‑generated text that evades a wide range of detectors.
**Method:** Detector‑guided beam search: at each generation step, top‑k candidate tokens are scored by a trained AI‑text detector; the token with the lowest AI‑score (most human‑like) is selected, producing adversarial paraphrases without additional model training.
**Datasets:** Not explicitly listed.
**Key results:** Reduces true‑positive rate at 1 % false‑positive (T@1%F) by 64.49 % on RADAR and 98.96 % on Fast‑DetectGPT; average reduction of 87.88 % across eight detectors; text quality remains high.
**Limitations:** Not fully detailed.

### 4. ’Your AI Text is not Mine’: Redefining and Evaluating AI‑generated Text Detection under Realistic Assumptions (uploaded‑paper: upload-sample-uploaded-paper)
**Problem:** Existing detection research lacks a shared definition of harmful AI‑generated text and ignores realistic human‑AI co‑creation.
**Method:** Proposes a taxonomy of detection notions; builds AITDNA benchmark (350+ co‑written texts from 99 authors with interaction logs); evaluates detectors against each notion.
**Datasets:** AITDNA.
**Key results:** Detectors perform well only for narrow notions; struggle with content‑ and authorship‑ID notions; prior benchmarks miss realistic co‑creation scenarios.
**Limitations:** Not fully detailed.

## Comparison Table

| Paper | Method | Datasets | Key Result | Limitations |
|---|---|---|---|---|
| **Agents Can Use Base Models to Evade AI Detection** (2609.31876v1) | HALO copies up to 90 % base LM tokens via `<copy>` tags; orchestrated agent assembles final response. | Mythos, FACTS, HealthBench, IFEval | Detection rate drops from 77 % to 24 %; maintains >90 % task performance; query cost ↑ up to 30×. | High computational/monetary cost; incomplete evasion on HealthBench; needs local base model. |
| **PADBen Benchmark** (2511.00416v1) | Constructs benchmark with five text types and five detection tasks; dual‑space analysis of paraphrase effects. | PADBen (350+ texts) | Detectors fall from >90 % to near‑random on iteratively paraphrased text; reveals laundering region. | Limited methodological detail in paper. |
| **Adversarial Paraphrasing** (2506.07001v2) | Detector‑guided beam search using off‑the‑shelf LLM; selects token with lowest AI‑score each step. | Not explicitly listed | Avg. T@1%F reduction 87.88 % across eight detectors; minimal quality loss. | Limited methodological detail in paper. |
| **Your AI Text is not Mine** (upload‑sample‑uploaded‑paper) | Taxonomy of detection notions; AITDNA benchmark of co‑written texts; notion‑wise evaluation. | AITDNA | Detectors excel only on narrow notions; struggle with content‑ and authorship‑ID notions; prior datasets unrealistic. | Limited methodological detail in paper. |

**Overall Insight**
- **Evasion Techniques:** HALO (token copying) and Adversarial Paraphrasing (detector‑guided token selection) achieve strong evasion, but HALO incurs high cost while Adversarial Paraphrasing is training‑free and broadly effective.
- **Evaluation Gaps:** PADBen provides a systematic benchmark exposing detector blind spots; AITDNA adds realism by modeling human‑AI co‑creation and shows notion‑specific performance variance.
- **Research Direction:** Future work should aim for cost‑effective evasion methods, develop detectors robust across the laundering region identified by PADBen, and adopt realistic evaluation frameworks like AITDNA that consider diverse detection notions.
