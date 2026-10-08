# Derived manuscript results

This directory contains the machine-readable outputs used to audit the final manuscript.

- `metrics_14_rules.csv` contains the 14 scoring rules used for the model-ranking analyses.
- `table_*.csv` files contain derived analyses from the same artifact lineage.
- `manuscript_tables/` contains compact CSV exports of the seven tables printed in the submission manuscript.
- `derived_nonidentification_all8.csv` contains the eight-benchmark non-identification construction outputs.

The released rule set uses the label `PFA-shared` for the constrained PFA-style baseline with a skill-specific intercept and shared prior-success/prior-failure slopes. This distinguishes the implemented baseline from canonical PFA with skill-specific success/failure slopes.

Raw third-party benchmark files and large cached prediction arrays are not committed. Use the repository data-retrieval script and analysis pipeline to regenerate them.
