# Build the CBM evaluation manuscript.
# Regenerate the tables first: every one is derived from report/results/
# and the prediction archives, so the paper's numbers follow the artefacts.
& "$env:USERPROFILE\miniconda3\envs\dl_env\python.exe" -m src.scripts.paperB_tables
latexmk -pdf -interaction=nonstopmode main
