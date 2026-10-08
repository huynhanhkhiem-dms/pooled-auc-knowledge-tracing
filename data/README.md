# Benchmark data

The manuscript evaluates eight public/preprocessed knowledge-tracing benchmarks:
ASSISTments 2009, 2012, 2015, and 2017; Algebra 2005; Bridge to Algebra 2006;
Spanish; and Statics.

The repository does **not** mirror the third-party benchmark files. Their original
provider terms remain in force. Run `./fetch_data.sh` from the repository root to
retrieve the pinned preprocessing release into `data/benchmarks/`.

The retrieval script pins
`theophilegervet/learner-performance-prediction` to commit
`a7ae193aa6957003a764aed7c95b07666fd4f1da`.

`source_manifest.csv` records the dataset-to-source mapping.
