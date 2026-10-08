# Pooled AUC Does Not Identify Within-Learner Predictive Discrimination in Knowledge Tracing

Reproducibility repository for the manuscript by **Huynh Anh Khiem**
(Ton Duc Thang University; ORCID 0009-0007-7210-174X).

## What is included

- synchronized analysis code for the **14 scoring rules** reported in the manuscript;
- the exact pooled/within/between AUC decomposition and nested learner-KC/item metrics;
- the canonical skill-specific PFA implementation used by the manuscript specification;
- bounded-context DKT and SAKT scoring implementations used only as heterogeneous score functions;
- causality, gradient, leakage, and decomposition tests;
- a pinned benchmark-data retrieval script and source manifest;
- machine-readable exports of the manuscript tables.

## Data

Third-party benchmark files are not mirrored in this repository. Their provider terms remain in force.
Run `./fetch_data.sh` to retrieve the public preprocessing release used by the analysis. The retrieval
is pinned to Gervet et al.'s preprocessing repository at commit
`a7ae193aa6957003a764aed7c95b07666fd4f1da`.

## Environment

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
chmod +x fetch_data.sh
./fetch_data.sh
```

## Correctness checks

```bash
python code/tests/test_gradients.py
python code/tests/test_sakt_gradients.py
python code/tests/test_leakage.py
```

All three test suites passed in the release prepared on 2026-10-08.

## Scoring rules

GlobalMean, ItemMean, SkillMean, FrozenAbility, FrozenIRT, RunningAbility, PFA,
BKT, Best-LR, Best-LR+I, DKT, DKT+I, SAKT, and SAKT+I.

PFA uses one skill-specific intercept and separate skill-specific coefficients for prior
successes and failures on the active skill. Outcome-dependent features are strictly causal.

## Full reproducibility package

The repository contains `reproducibility_package.zip`, which includes the synchronized source
code, tests, data-source manifest, data-retrieval/build scripts, and machine-readable manuscript tables.
Large cached prediction arrays are regenerated locally and are not committed.

## License

Analysis code: MIT License. Benchmark licenses and original provider terms remain unchanged.
