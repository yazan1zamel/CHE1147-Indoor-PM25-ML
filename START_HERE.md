# 🚀 Uploading the complete research repository

This folder is the **complete local repository content**. It contains **all 267 original files unchanged**, including raw research data, plus new organization and the **Second_Model** folder.

## Recommended: GitHub Desktop (Mac)

GitHub browser uploads permit **up to 100 files per batch**. The raw dataset in this package contains hundreds of files, so GitHub Desktop is simpler.

**If you have already created an empty online repo:**
1. Install [GitHub Desktop](https://desktop.github.com/), sign in to your GitHub account.
2. In GitHub Desktop choose **File → Clone repository** and select your new empty `CHE1147-Indoor-PM25-ML` repo.
3. Open its new local folder in Finder, copy **everything inside this package's `CHE1147-Indoor-PM25-ML` folder** into that cloned folder (including hidden `.gitignore`).
4. Return to GitHub Desktop: review changed files, enter a commit summary such as `Add original raw data and Second Model results`, click **Commit to main**, then click **Push origin**.
5. Refresh the repository's GitHub website to see the formatted README.

**If you have NOT created the online repo yet:** GitHub Desktop → File → New repository (choose the extracted folder's location, or make a new clean GitHub Desktop repository and copy these files there), commit and **Publish repository**. Choose **Private** during testing.

## Safety and data integrity

- All original files are **byte-identical** to those in the team ZIP. See [`PRESERVATION_AUDIT.md`](PRESERVATION_AUDIT.md).
- The original top-level `.gitignore` was **not modified**. Additional ignore overrides inside `DATA/` and `Second_Model/` ensure the raw CSV files and results can be uploaded.
- The dataset is approximately **260 MB uncompressed**; GitHub Desktop may require some time to push. Do not upload the ZIP itself to GitHub.
- **Private first:** verify dataset redistribution rights and get team approval before sharing the full raw data publicly.
- The HTML results dashboard is self-contained. Download [`Second_Model/results_dashboard.html`](Second_Model/results_dashboard.html) and open locally, or configure GitHub Pages where allowed.
