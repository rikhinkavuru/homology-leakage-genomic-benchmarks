# Reconciled headline numbers (T1.8): decision-rule + estimator audit

Scope: this file only *documents* the reconciliation. It changes no frozen result,
no other file, and no committed CSV. It resolves why the manuscript carries **three
coexisting headline accuracy-drop values per leaky dataset** (ensembl 15.6 / 16.0 /
16.4 pts; nonTATA 10.8 / 12.1 / 12.2 pts) and recommends one canonical convention.

---

## 1. Decision-rule inventory (every accuracy computation)

Two decision rules coexist in the codebase. `.predict()` is sklearn argmax
(`classes_[argmax(predict_proba)]`, i.e. `proba[:,1] > 0.5`); the other is an
explicit `predict_proba(...)[:,1] >= 0.5` threshold. For binary LR/RF they differ
*only* on exact 0.5 ties (RF 75/150 vote splits): `.predict()` breaks the tie to
class 0, `>=0.5` to class 1.

| file:line | code | rule |
|---|---|---|
| `run_audit.py:157-158` | `proba = model.predict_proba(...)[:,1]` ; `pred = (proba >= 0.5).astype(int)` | **predict_proba>=0.5** (this is `RA.eval_split`) |
| `run_suite.py:249` | `evalf = eval_split_multi if multiclass else RA.eval_split` (binary → `RA.eval_split`) | **predict_proba>=0.5** |
| `run_suite.py:205-206` | `eval_split_multi`: `pred = model.classes_[np.argmax(proba, axis=1)]` | argmax (=`.predict()`), **multiclass only** |
| `run_fullscale.py:51` | `cfg = S.eval_all_configs(X, y, tr, te, RA.eval_split)` | **predict_proba>=0.5** |
| `run_robustness_full.py:45` | `cfg = S.eval_all_configs(X, y, tr, te, RA.eval_split)` | **predict_proba>=0.5** |
| `run_extended_models.py:53` | `pred = model.predict(X[te])` | **.predict()** |
| `step_variance_ci.py:59` | `return (model.predict(X[te]) == y[te])` | **.predict()** |
| `step2_rf_seeds.py:36` | `accs.append(float((m.predict(X[te]) == y[te]).mean()))` | **.predict()** (produces the RF_k6 rows of `step2_seed_variance.csv`) |
| `rf_seed_variance.py:54` | `return float((model.predict(X[te]) == y[te]).mean())` | **.predict()** |
| `expkit.py:200, 206` | `correctness`/`metrics`: `model.predict(X[te])` | **.predict()** (declared canonical) |

`RA.eval_split` (`run_audit.py:153-165`) verbatim:

```python
def eval_split(X, y, train_idx, test_idx):
    out = {}
    for name, model in make_models().items():
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)          # <-- predict_proba>=0.5 rule
        yt = y[test_idx]
        out[name] = dict(accuracy=accuracy_score(yt, pred), ...)
```

**Two pipelines, two rules.** The *main-results / robustness* pipeline
(`run_audit` → `run_suite` → `run_fullscale` → `run_robustness_full`) uses
`predict_proba>=0.5`. The *variance / CI / extended / rf-seed* pipeline
(`step_variance_ci`, `step2_rf_seeds`, `rf_seed_variance`, `run_extended_models`,
`expkit`) uses `.predict()`.

---

## 2. Provenance of each of the three numbers (RF k=6, confirmed from CSVs)

| dataset | drop (pts) | estimator | decision rule | original acc | corrected acc | source CSV |
|---|---|---|---|---|---|---|
| nonTATA | **12.1** (0.1212) | 3-seed frozen mean | predict_proba>=0.5 | 0.9317 | 0.8105 | `fullscale_long.csv` (seeds 0/1/2 = 0.8186/0.8089/0.8041) → `fullscale_summary.csv` |
| nonTATA | **12.2** (0.1216) | 5-seed mean | .predict() (corr) | 0.9317 (proba) | 0.8101 | `step2_seed_variance.csv` (mean 0.8101) − frozen orig 0.9317 |
| nonTATA | **10.8** (0.1083) | seed-0 point + bootstrap | .predict() | 0.9284 | 0.8200 | `step3_delta_ci.csv` (0.1083 [0.0993,0.1183]) = 0.9284−0.8200 |
| ensembl | **15.6** (0.1558) | 3-seed frozen mean | predict_proba>=0.5 | 0.8556 | 0.6999 | `fullscale_long.csv` (seeds 0/1/2 = 0.6999/0.7023/0.6973) → `fullscale_summary.csv` |
| ensembl | **16.0** (0.1605) | 5-seed mean | .predict() (corr) | 0.8556 (proba) | 0.6951 | `step2_seed_variance.csv` (mean 0.6951) − frozen orig 0.8556 |
| ensembl | **16.4** (0.1639) | seed-0 point + bootstrap | .predict() | 0.8596 | 0.6957 | `step3_delta_ci.csv` (0.1639 [0.1579,0.1704]) = 0.8596−0.6957 |

Notes confirmed from the CSVs:
- The **original accuracy itself changes with the rule**: nonTATA RF k6 = 0.9317
  (`>=0.5`) vs 0.9284 (`.predict()`); ensembl = 0.8556 (`>=0.5`) vs 0.8596
  (`.predict()`). Opposite signs, because the tied examples' true labels skew
  differently per dataset.
- Only the **seed-0 headline (10.8 / 16.4)** is rule-consistent end-to-end
  (`.predict()` for both original and corrected). The 3-seed (12.1 / 15.6) is
  `>=0.5` throughout; the 5-seed (12.2 / 16.0) mixes a `>=0.5` original with a
  `.predict()` corrected mean.

---

## 3. Do the two rules actually differ? (measured, cached nonTATA, full scale)

Fit RF k6 (`random_state=0`, identical to frozen), accuracy computed BOTH ways
(`scratchpad/verify_rules.py` via `expkit`):

| split | `.predict()` (argmax) | `proba>=0.5` | rule Δ(acc) | exact-0.5 ties | examples flipped |
|---|---|---|---|---|---|
| original | 0.9284 | 0.9317 | **+0.0033** | 40 | 40 (all) |
| corrected seed-0 | 0.8200 | 0.8186 | **−0.0014** | 87 | 87 (all) |

Seed-0 drop: `.predict()` = **10.83 pts**, `proba>=0.5` = **11.31 pts**
→ rule-induced difference in the *drop* = **0.48 pts**.

**The rules do differ, but the difference is tiny and fully deterministic.** Every
flipped example is a 75/150 RF vote-tie broken oppositely by the two rules
(0.4–1.0 % of the test set). RF with a fixed `random_state` is bit-deterministic
regardless of `n_jobs`, so this is *deterministic tie-breaking*, **not**
"RandomForest thread-nondeterminism."

**Consequence for the 3-number spread.** The nonTATA spread (10.8→12.2 = 1.4 pts)
is *dominated by the estimator choice*, not the decision rule: seed-0 corrected
(0.8200) is the luckiest of the 5 partitions (5-seed range [0.8036, 0.8200],
mean 0.8101), so using the seed-0 point instead of the mean shrinks the drop by
~1 pt, while the decision rule contributes only ~0.3–0.5 pt. The ensembl spread
(15.6→16.4 = 0.8 pts) is a roughly even mix of the rule (~0.4 pt on the original)
and the 3-seed-vs-5-seed corrected estimate (~0.5 pt); the corrected accuracy is
otherwise seed-stable (SD 0.0019).

---

## 4. True cause of the three coexisting numbers

The three values per dataset are three *post-hoc summaries*, each computed with a
different combination of (a) decision rule and (b) corrected-accuracy estimator:

1. **S3/S5 frozen tables** — 3-seed-mean corrected, `predict_proba>=0.5`
   → 12.1 / 15.6.
2. **S16.1 seed variance** — 5-seed-mean corrected `.predict()`, minus the
   frozen `>=0.5` original → 12.2 / 16.0.
3. **S16.2 headline** — seed-0 point corrected, `.predict()`, bootstrap CI
   → 10.8 / 16.4.

So the spread is a joint artifact of **decision-rule inconsistency** (small,
deterministic, RF vote-ties) **and estimator inconsistency** (seed-0 point vs
3-seed-mean vs 5-seed-mean, with mismatched original baselines 0.9317 vs 0.9284 /
0.8556 vs 0.8596). **Neither is thread-nondeterminism.** The manuscript's hidden
LaTeX comment (`paper/main.tex:42-43`) — *"these differ by RandomForest
thread-nondeterminism (~0.004) and the seed-0 vs mean corrected split"* —
mislabels the deterministic ~0.003–0.004 tie-break as thread-nondeterminism, and
under-weights the seed-0-vs-mean estimator choice, which is the larger driver on
nonTATA.

Cherry-pick note: the headline reports ensembl **16.4** (the *max* of its three)
and nonTATA **10.8** (the *min* of its three). Both are the same estimator (seed-0
`.predict()` bootstrap), so it is uniform in *method*, but seed-0 is a single
arbitrary partition that happens to land at opposite extremes of the two datasets'
5-seed ranges — which reads as opposite-direction cherry-picking.

---

## 5. Recommended canonical convention + reconciled numbers

**Convention (apply uniformly to BOTH datasets):**
- **Decision rule:** `.predict()` (argmax) everywhere — already declared canonical
  in `expkit.py`.
- **Estimator:** **5-seed-mean corrected accuracy** (does not depend on the
  arbitrary seed-0 partition), original from the single fixed benchmark split,
  drop = original − 5-seed-mean corrected, with the test-set bootstrap 95% CI for
  sampling uncertainty and the 5-seed SD for partition uncertainty.

**Reconciled headline drops (all values from committed CSVs):**

| dataset | original acc (.predict, `step3_accuracy_ci.csv` / `rf_seed_variance.csv`) | corrected 5-seed mean±SD (.predict, `step2_seed_variance.csv`) | **reconciled drop** | bootstrap 95% CI (`step3_delta_ci.csv`) |
|---|---|---|---|---|
| human_nontata_promoters (RF k6) | 0.9284 | 0.8101 ± 0.0055 | **11.8 pts** (0.1183) | [0.099, 0.118] |
| human_enhancers_ensembl (RF k6) | 0.8596 | 0.6951 ± 0.0019 | **16.5 pts** (0.1645) | [0.158, 0.170] |

Both drops remain highly significant (bootstrap CIs exclude 0). These replace the
seed-0-point pair (10.8 / 16.4), removing the "seed-0 is lucky in opposite
directions" fragility while keeping a single decision rule and a single estimator.

*(If instead the existing seed-0 bootstrap machinery is kept verbatim, the uniform
`.predict()` seed-0 pair is 10.8 / 16.4 — already the headline — and the only
remaining fix is deleting the incorrect "thread-nondeterminism" wording.)*

---

## 6. The one-line code fix

Make every path use `.predict()`. In `RA.eval_split` (imported by `run_suite`,
`run_fullscale`, `run_robustness_full`), replace the threshold with argmax:

**`run_audit.py:158`**
```python
-        pred = (proba >= 0.5).astype(int)
+        pred = model.predict(X[test_idx])
```

Keep line 157 (`proba = model.predict_proba(...)[:,1]`) for AUROC. This single edit
harmonizes the main-results/robustness pipeline with the variance/CI pipeline, so
`fullscale_long.csv`, `fullscale_summary.csv`, and `robustness_fullscale_summary.csv`
regenerate on the `.predict()` rule and the 12.1/15.6 numbers collapse onto the
`.predict()` estimates. (Then also correct `paper/main.tex:42-43`: the residual
~0.003–0.004 is deterministic RF tie-breaking, not thread-nondeterminism.)
```
