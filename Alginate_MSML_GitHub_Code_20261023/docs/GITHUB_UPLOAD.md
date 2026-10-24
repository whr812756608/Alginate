# Publish the repository to GitHub

An existing destination was found: `https://github.com/whr812756608/Alginate` (**PUBLIC**, empty as inspected on 2026-10-20). **Confirm public-release permission from the coauthors before uploading** any project code or analysis results.

If that is the desired destination, from inside the unzipped `Alginate_MSML_GitHub_Submission_20261022` directory run:

```bash
git init -b main
git add .
git status
git commit -m "Add NMR-calibrated alginate MS-ML analysis and reproducibility package"
git remote add origin https://github.com/whr812756608/Alginate.git
git push -u origin main
```

For a private release, first create or switch to a **private** GitHub repository and update the remote URL. If the destination now has commits, clone it first and copy the package contents into the clone before committing, instead of force-pushing.

The package intentionally does not include raw spectra or unpublished Word manuscripts. **GitHub visibility is not controlled by `.gitignore`**: excluded data must also stay outside the folder when publishing. The synthetic integration tests execute in GitHub Actions and are not substitutes for experimental validation.

The updated package includes a labeled **synthetic** Figure 3c preview, plus actual Figure 3e/f SVGs and supplied V7.0 analytical tables. DO NOT include the synthetic preview as measured source data in the manuscript. To publish a real 3c, rerun the script on approved mzML data and archive its exported source CSV.
