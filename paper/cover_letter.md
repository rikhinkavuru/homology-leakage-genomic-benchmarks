# Cover letter — resubmission

To: Dr. Lina Ma, Editor-in-Chief, and the Editorial Office\
Bioinformatics Advances

Dear Dr. Ma and Editors,

Please find enclosed a thoroughly revised manuscript, **"Near-duplicate leakage can reorder model rankings on a genomic benchmark suite: an audit of Genomic Benchmarks."** This is a resubmission of manuscript **BIOADV-2026-296** (previously titled *"Homology leakage selectively changes which model wins on genomic sequence-classification benchmarks,"* Associate Editor: Prof. Shanfeng Zhu), which received a Reject & Revise decision. As instructed, we submit it as new work and reference the prior manuscript number here so the records can be linked; a point-by-point response to every reviewer and Associate-Editor comment is included as a separate document.

We took the reviewers' and Associate Editor's concerns as a mandate to test the paper's central claim far harder than the original submission did, and the revision changes both what we claim and how strongly. In brief:

- **Deep-learning generality (R1.1, R2.a3, R3.1).** We add a from-scratch 1D residual CNN with a pre-registered dropout×weight-decay dose-response and a binding refutation condition; its regularized reference configuration reproduces the leakage-driven accuracy drop on both leaky datasets (cluster-bootstrap CIs excluding zero), so the effect is not a classical-model artifact. The specific pretrained models Reviewer 3 named (DNABERT-2, HyenaDNA, the Nucleotide Transformer) are pretrained on the human reference genome from which these test sequences are drawn — an ~100% pretraining overlap we now quantify — so fine-tuning them cannot cleanly isolate the train/test split effect; we scope them out, name and measure that second leakage channel, and leave the deployed-foundation-model question explicitly open.

- **Homology-detection validation (R2.a2, R1.3, R3.4).** We validate the leakage with alignment identity (MMseqs2), which reproduces the effect metric-independently, and add a length-robust containment index that reveals leakage the k-mer Jaccard misses. Throughout, we adopt the more accurate term **near-duplicate leakage** and reserve "homology" for where alignment/containment evidence earns it.

- **Bootstrap methodology (R3.3).** We add a cluster (block) bootstrap over within-test near-duplicate components, with intraclass correlation, design effect, an analytic cluster-robust cross-check, and a combined-source interval; no significance verdict changes.

- **Dataset characteristics and hyperparameters (R2.a1, R3.2).** We disclose that the balanced datasets are curated (not naturally balanced) and add a prevalence-aware imbalanced evaluation; and we show via a random-forest regularization path that the effect is a property of the *default unregularized* forest, dissolving under regularization.

- **Honest rescoping.** Most importantly, we now demonstrate the ranking-change claim **cleanly on one dataset** (human_enhancers_ensembl — airtight under accuracy, AUROC, F1, a chromosome-holdout control, a winner-probability bootstrap, and alignment) and present the second (human_nontata_promoters) as an explicit **cautionary partial case**, together with a discussion of the trade-off that de-duplication can remove learnable signal (R1.2). The title, abstract, and claims have been revised accordingly.

The study remains fully reproducible; all code, the report card, and the near-duplicate-aware splitter are available in the linked repository. The work is original, is not under consideration elsewhere, and all authors approve this submission. We have no competing interests to declare.

We are grateful to the reviewers and the Associate Editor — their comments materially improved the paper — and we hope the revised manuscript now meets the standard for publication.

Sincerely,\
Rikhin Kavuru\
Independent Researcher\
rikhinkavuru@gmail.com
