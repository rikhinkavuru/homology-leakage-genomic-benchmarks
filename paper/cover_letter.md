# Cover letter — resubmission

To: Dr. Lina Ma, Editor-in-Chief, and the Editorial Office\
Bioinformatics Advances

Dear Dr. Ma and Editors,

Please find enclosed a thoroughly revised manuscript, **"Near-duplicate leakage can reorder model rankings on a genomic benchmark suite: an audit of Genomic Benchmarks."** This is a resubmission of manuscript **BIOADV-2026-296** (previously titled *"Homology leakage selectively changes which model wins on genomic sequence-classification benchmarks,"* Associate Editor: Prof. Shanfeng Zhu), which received a Reject & Revise decision. As instructed, we submit it as new work and reference the prior manuscript number here so the records can be linked; a point-by-point response to every reviewer and Associate-Editor comment is included as a separate document.

The decision turned on a single question — whether the ranking result is general or an artifact of one narrow comparison — and we treated it as a mandate to test the paper's central claim far harder than the original submission did. Four additions carry the revision:

- **A nine-model roster (§4.4).** The strongest form of the objection is that a claim about "rankings" made over four models is a claim about nothing. We widened the comparison to nine learners spanning memorization propensity end to end. The result is both stronger and sharper than before: the leaky split **cannot separate its top three models at all** — they finish within 0.0015 of one another — **yet those same three span 23 accuracy points once the split is corrected**, and 1-nearest-neighbour falls from rank 2 to last, 0.873 to 0.537. The benchmark is not merely crowning the wrong model; it is blind to large real differences between models it reports as tied.

- **A construct-and-break manipulation (§4.7).** The previous submission argued the cause was a curation defect but did not test it. We now intervene on the construction step alone, in both directions: applying the omitted merge step to the leaky dataset abolishes the reordering, and imposing the same defect on a clean dataset — at matched size and balance, with model code and sequences untouched — manufactures it on demand. This makes the causal claim interventional rather than correlational.

- **A cross-suite census, and an honest null (§4.10).** Applying the census unchanged to the Nucleotide Transformer downstream tasks finds three of twelve independent tasks leaky, one at 25% byte-identical overlap. **No ranking inverts materially there, and we report that null prominently.** (One swap does occur, but the two models were separated by only 0.005 on the as-shipped split, so it is immaterial under our own criterion — we state it either way.) It is a test passed rather than an embarrassment: our diagnostic condition predicts the null in advance from the as-shipped split alone, because reordering requires not just leakage but a challenger genuinely better on novel sequences — a condition this second suite does not supply.

- **Honest rescoping.** We now demonstrate the ranking-change claim **cleanly on one dataset** (`human_enhancers_ensembl`, holding under accuracy, AUROC, F1, a chromosome-holdout control, an alignment-scored re-measure and a bootstrap winner probability) and present the second (`human_nontata_promoters`) as an explicit **cautionary partial case**, with five independent signals showing why. The title, abstract, Discussion and new Conclusion all carry that scope.

The revision also answers the specific methodological requests: a from-scratch 1D CNN with a pre-registered grid and a binding refutation condition (the named pretrained models are scoped out with a *quantified* ≈100% pretraining-overlap confound, not an asserted one); MMseqs2 alignment validation and a length-robust containment index, which honestly flags one previously-clean dataset as borderline; a cluster bootstrap with intraclass correlation, design effect, an analytic cluster-robust cross-check and a combined-source interval; a prevalence-aware imbalanced evaluation; and a random-forest regularization path. On that last point we ran the experiment the objection really demands — whether ordinary cross-validation would rescue a practitioner — and found it does not: both naive and cluster-grouped cross-validation select the memorizing default. Our own pre-registered expectation to the contrary was wrong, and we report it as a failure.

We have tried throughout to report what did not work as prominently as what did: a pre-registered dose-response that turned out non-monotone; one pre-registered prediction (P5) failing outright and two more holding only conditionally, all five scored in the text whether or not they held; a cross-suite null; and a tuning experiment we could afford on only one of the two leaky datasets — and it is `human_nontata_promoters`, the dataset on which we *decline* to claim the reordering, which we state plainly rather than letting the result stand for both. Where evidence is slender we say so rather than presenting it as more.

The study remains fully reproducible; all code, the report card, and the near-duplicate-aware splitter and certification tool are available in the linked repository, and the manuscript now states its pinned software environment. The work is original, is not under consideration elsewhere, and all authors approve this submission. We have no competing interests to declare.

We are grateful to the reviewers and the Associate Editor — their comments materially improved the paper — and we hope the revised manuscript now meets the standard for publication.

Sincerely,\
Rikhin Kavuru\
Independent Researcher\
rikhinkavuru@gmail.com
