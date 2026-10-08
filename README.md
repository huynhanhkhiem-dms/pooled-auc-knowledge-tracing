# Pooled AUC Does Not Identify Within-Learner Predictive Discrimination in Knowledge Tracing

Public companion repository for the manuscript by **Huynh Anh Khiem**
(Ton Duc Thang University; ORCID 0009-0007-7210-174X).

## Scope of this repository

This repository exposes the code and machine-readable artifacts needed to audit the
paper's central methodological claims:

- exact pooled / within-learner / between-learner AUC decomposition;
- nested within-learner-within-KC and within-learner-within-item pair weighting;
- the rank-preserving non-identification construction;
- synthetic correctness tests for the decomposition identities and construction;
- a pinned benchmark-data retrieval script and source manifest;
- machine-readable exports of the seven tables reported in the manuscript.

The repository does **not** claim to mirror every third-party benchmark file or every
large cached model prediction used during the empirical study.

## Environment

Python 3.11+ is recommended.

```bash
python -m pip install -r requirements.txt
```

## Core correctness check

From the repository root:

```bash
python code/tests/test_core.py
```

Expected output:

```
ALL CORE CHECKS PASSED
```

## Benchmark data

The study uses eight public/preprocessed knowledge-tracing benchmarks:
ASSISTments 2009, 2012, 2015, and 2017; Algebra 2005; Bridge to Algebra 2006;
Spanish; and Statics.

Third-party benchmark files are not re-hosted here. Run:

```bash
chmod +x fetch_data.sh
./fetch_data.sh
```

The script retrieves the preprocessing release pinned to commit
`a7ae193aa6957003a764aed7c95b07666fd4f1da` of
`theophilegervet/learner-performance-prediction`. Original provider terms remain
in force.

## Reported results

`results/manuscript_tables/` contains CSV exports of the seven tables in the
submission manuscript. They are provided for transparent inspection and should be
read together with the manuscript definitions, estimands, and limitations.

## Code map

- `code/auc_metrics.py`: exact pair-count AUC decomposition and nested metrics.
- `code/nonidentification.py`: within-learner-rank-preserving restratification.
- `code/tests/test_core.py`: synthetic identity and non-identification checks.
- `data/source_manifest.csv`: pinned sources for the eight benchmark datasets.

## Reproducibility boundary

The public release is intentionally scoped to the paper's central metric and
identification contribution plus the reported table exports. Raw third-party
benchmarks and large cached prediction arrays are not redistributed.

## Citation

Please cite the associated manuscript. Machine-readable citation metadata are in
`CITATION.cff`.

## License

Repository analysis code is released under the MIT License. Dataset licenses and
original provider terms are unchanged.
