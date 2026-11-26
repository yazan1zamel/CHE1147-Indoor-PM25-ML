# 🗂 Data guide

The **original `DATA/` folder is included in this repository unchanged** from the provided team archive, including air-quality monitoring, airflow, and monitoring-state data. It is approximately 260 MB uncompressed. The original scripts use that directory.

The underlying research dataset is described in [the Dryad record](https://doi.org/10.7941/D1ZS7X). Check the dataset's terms and your team's permission **before making this repository public**. Prefer **Private** while testing.

The paper-aligned notebook's first setup cell has a `USER_DATA_ROOT` option. Set it to the absolute path of the repository's `DATA` folder, or to your local downloaded dataset. The notebook's original author-specific fallback path may not match another person's computer.

*Git ignore note:* The original top-level `.gitignore` is byte-for-byte unchanged and contains `*.csv` and `*.txt`. The additional `DATA/.gitignore` is deliberately configured to allow raw CSV/TXT files to be committed. GitHub Desktop is recommended because the raw dataset contains more than 100 files.
