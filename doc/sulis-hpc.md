# Running MSAGAT-Net on Sulis

Setup, transfer, run, monitoring and copy-back for the Sulis Tier-2 cluster
(login.sulis.ac.uk, user `ae6780`). Scripts live in `hpc/sulis/`. Facts about
Sulis are from https://sulis-hpc.github.io/ and from the cluster itself
(29 Sep 2026).

## 1. What the project needs (measured, not assumed)

| Item | Value | Source |
|---|---|---|
| Entry points | `python -m src.train --single ...` (one run); `python -m src.scripts.campaign --chunk <name>` (MSAGAT-Net grids); `python -m src.scripts.baseline_campaign` (5 baselines in `../colagnn`, `../EpiGNN`) | `src/` |
| Python / deps | Python 3.11; torch >= 2.0, numpy, pandas, scipy, scikit-learn, matplotlib, seaborn; baselines also import `tensorboardX` | imports in `src/`, `../colagnn`, `../EpiGNN` |
| Data | `data/` 6.7 MB, git-tracked; byte-identical copies in `../colagnn/data` and `../EpiGNN/data` | sha256, 29 Sep audit |
| Model size | 7k-40k parameters; GPU memory well under 2 GB per run | `report/manifests/**` |
| MSAGAT-Net run time (RTX 5060 Ti) | mean 45 s Japan, 56 s NHS, 93 s LTLA; max 381 s | 285 manifests |
| Baseline run time | CNNRNN-Res/LSTNet/EpiGNN 30-280 s; cola_gnn on LTLA mean 4.5 h, max 10.3 h; dcrnn 1-2 h (region785 up to 6.5 h) | `report/logs/baseline_campaign.log` |
| Totals so far | ~17 GPU-h MSAGAT-Net + ~205 GPU-h baselines, all serial on one shared card | campaign logs |

Consequence: an A100 is far more than one run needs. The work is many small
jobs, so the efficient pattern is **several runs sharing one GPU**
(`baseline_campaign.py --parallel`), not bigger GPUs. `campaign.py` currently
runs MSAGAT-Net one process at a time.

## 2. Layout

Sulis (GPFS home, **no backup**):

```
~/msagat/                      MSAGAT_ROOT
  MSAGAT-Net/                  git checkout, branch paper-b
    hpc/sulis/*.out            Slurm logs (git-ignored)
    save_all/ report/predictions/ tensorboard/   outputs (git-ignored)
  colagnn/                     code + data rsync'd from the laptop (the lead-h
  EpiGNN/                      protocol fix is uncommitted there; see §4)
  venv/                        project venv on top of Sulis modules
  venv-freeze.txt
```

`baseline_campaign.py` finds the baselines as siblings of `MSAGAT-Net`, so
keep all three directories under the same parent.

Laptop: results copied back land in
`C:\Users\ajaoo\Documents\GitHub\sulis-results\<date>\`, never directly over
the local `report/`, so they can be compared before anything is merged.

## 3. SSH (one 2FA prompt per 8 hours)

Sulis wants an SSH key **protected by a passphrase** plus a TOTP code.
`~/.ssh/id_ed25519` currently has no passphrase; add one:

```powershell
# PowerShell (Windows)
ssh-keygen -p -f $HOME\.ssh\id_ed25519
wsl -d Ubuntu -- ssh-keygen -p -f ~/.ssh/id_ed25519_sulis
```

WSL's OpenSSH shares one authenticated connection (`ControlMaster`); Git Bash's
and Windows' OpenSSH cannot. The `sulis` host in WSL's `~/.ssh/config` is set
up for this (ControlPersist 8h). Open the shared connection once:

```powershell
wsl -d Ubuntu -- ssh sulis "hostname; account-balance"
```

Type the 6-digit code when asked. For the next 8 hours `wsl -d Ubuntu -- ssh
sulis ...`, `rsync` and `scp` reuse it without asking again. Close it early with
`wsl -d Ubuntu -- ssh -O exit sulis`.

## 4. Transfer

The GitHub repo is public, so Sulis clones it without credentials. Push from
the laptop, pull on Sulis:

```powershell
# PowerShell, in the MSAGAT-Net checkout
git push origin paper-b
git log -1 --oneline
```

```bash
# Sulis (first time)
mkdir -p ~/msagat && cd ~/msagat
git clone -b paper-b https://github.com/michaelajao/MSAGAT-Net.git
# Sulis (afterwards)
cd ~/msagat/MSAGAT-Net && git pull --ff-only
git log -1 --oneline                        # must match the laptop
```

For commits you don't want public yet, ship them as a bundle instead:
`git bundle create $env:TEMP\x.bundle origin/paper-b..paper-b`, copy it with
`wsl -d Ubuntu -- scp /mnt/c/Users/ajaoo/AppData/Local/Temp/x.bundle sulis:msagat/`,
then on Sulis `git pull --ff-only ~/msagat/x.bundle paper-b`.

Files that are **not** in git and must be copied: `AGENTS.md`/`CLAUDE.md`
(optional) and the two baseline repos. Their lead-h fix exists only as local
edits, so copy the working trees, code and data only:

```powershell
# PowerShell
foreach ($r in 'colagnn','EpiGNN') {
  wsl -d Ubuntu -- rsync -av --delete `
    --exclude save/ --exclude result/ --exclude tensorboard/ `
    --exclude __pycache__/ --exclude '*.pt' --exclude '*.pth' --exclude .git/ `
    "/mnt/c/Users/ajaoo/Documents/GitHub/$r/" "sulis:msagat/$r/"
}
```

Integrity: the data are checked on Sulis by `check.slurm` (sha256 of every
`data/*.txt` must match across the three repos). For the code, compare
`git rev-parse HEAD` on both sides, and for the baselines:

```powershell
wsl -d Ubuntu -- bash -c "cd /mnt/c/Users/ajaoo/Documents/GitHub/colagnn/src && sha256sum *.py" > $env:TEMP\colagnn.sha
wsl -d Ubuntu -- bash -c "cat /mnt/c/Users/ajaoo/AppData/Local/Temp/colagnn.sha | ssh sulis 'cd msagat/colagnn/src && sha256sum -c --quiet' && echo colagnn OK"
```

## 5. Environment (once)

```bash
# Sulis login node
account-balance                  # confirm su003-csmm has GPU-hours before any job
mmlsquota --block-size auto      # home quota: 2 TB / 2M files
sinfo -s                         # partitions: gpu, gpu-devel, ...
module spider PIP-PyTorch        # confirm 2.4.0-CUDA-12.4.0 still exists
module spider PIP-PyTorch/2.4.0-CUDA-12.4.0   # shows prerequisites (GCC/13.2.0 OpenMPI/4.1.6)
module spider SciPy-bundle/2023.11
bash ~/msagat/MSAGAT-Net/hpc/sulis/setup_venv.sh
```

Measured on 29 Sep 2026: `su003-csmm` held 10,947 GPU-hours and 9,933
CPU-hours (`su003` and `su003-gpulowpri` have no GPU-hours); home
`/home/a/ae6780` used 35 GB of 2 TB. The module stack loads cleanly and gives
Python 3.11.5, torch 2.4.0+cu124, numpy 1.26.2, scipy 1.11.4, pandas 2.1.3;
scikit-learn, matplotlib and the TensorBoard packages come from pip.
`setup_venv.sh` pins the module versions so pip cannot replace them.

`hpc/sulis/env.sh` loads the modules **then** activates the venv; every job
sources it. Conda is not used (Sulis advises against it). Results so far were
produced with torch 2.7 / pandas 3.0 on Windows; `check.slurm` measures
whether the Sulis stack changes anything.

## 6. Short check (gpu-devel, <= 50 min, <= 0.84 A100-hours)

```bash
cd ~/msagat/MSAGAT-Net
sbatch hpc/sulis/check.slurm
squeue -u $USER
tail -f hpc/sulis/msagat-check-<jobid>.out
```

It verifies imports and CUDA, data identity across repos, the test suite (930
tests locally), then re-runs one LTLA cell whose local result is on record
(RMSE 87.6404, 134.6 s) and prints the Sulis RMSE and wall time next to it, and
times one run on CPU only. Tracked result files it overwrites are restored at
the end. Use its wall time to cost any campaign:
`GPU-hours ≈ runs × wall_seconds / 3600` for serial MSAGAT-Net chunks.

## 7. Campaigns

```bash
# dry run on the login node (no GPU needed; prints what would run)
source hpc/sulis/env.sh
python -m src.scripts.campaign --chunk v2_main --dry-run | head
python -m src.scripts.baseline_campaign --dry-run --models lstnet

# submit
sbatch hpc/sulis/campaign.slurm v2_main
sbatch hpc/sulis/baselines.slurm --models lstnet CNNRNN_Res epignn
PARALLEL=3 sbatch --time=24:00:00 hpc/sulis/baselines.slurm --models cola_gnn
```

All jobs use `--account=su003-csmm`, `--gres=gpu:ampere_a100:1`, 42 CPUs,
3850 MB/CPU (the Sulis per-GPU share). For an L40 instead, pass
`--gres=gpu:lovelace_l40:1` (both types are in `gpu` and `gpu-devel`, three per
node; confirmed with `sinfo -o "%P %G"`).
GPU partitions are charged in GPU-hours; the CPUs requested with the GPU are
not charged separately. `gpu` allows 48 h per job; `gpu-devel` 1 h.

**Resume**: both drivers skip runs whose `report/predictions/**.npz` exists, so
resubmit the same command after a timeout or failure.

## 8. Monitoring and accounting

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,Partition,Elapsed,State,ExitCode,MaxRSS,AllocTRES%60
sacct -u $USER -S 2026-09-29 --format=JobID,JobName,Elapsed,State,AllocTRES%60
account-balance
tail -n 50 report/logs/campaign_<chunk>.log     # one line per run: token, ok/FAIL, seconds
```

## 9. Copy results back (GPFS home has no backup)

After every campaign, copy the cited artefacts plus logs to the laptop:

```powershell
$d = Get-Date -Format yyyy-MM-dd
wsl -d Ubuntu -- rsync -av `
  --include '*/' --include 'report/results/**' --include 'report/manifests/**' `
  --include 'report/predictions/**' --include 'report/logs/**' `
  --include 'hpc/sulis/*.out' --exclude '*' `
  sulis:msagat/MSAGAT-Net/ "/mnt/c/Users/ajaoo/Documents/GitHub/sulis-results/$d/"
```

Checkpoints (`save_all/`, ~0.7 MB each) only if a re-evaluation will need them:
add `--include 'save_all/**'`. Record the Sulis commit (`git rev-parse HEAD`),
the job IDs and the `venv-freeze.txt` alongside each copy.

## 10. VS Code and interactive GPU work

VS Code Remote-SSH connected to `sulis` edits files on the **login node**, which
has no GPU. Set it up so it reuses the WSL connection, or use Remote-SSH's own
connection (it prompts for the TOTP in the VS Code terminal panel):

1. Install "Remote - SSH"; add to `C:\Users\ajaoo\.ssh\config`:
   `Host sulis-vscode` / `HostName login.sulis.ac.uk` / `User ae6780` /
   `IdentityFile ~/.ssh/id_ed25519`.
2. Connect, open `~/msagat/MSAGAT-Net`.
3. Python interpreter: *Python: Select Interpreter* → *Enter path* →
   `~/msagat/venv/bin/python`. Because modules must be loaded first, run
   commands in an integrated terminal after `source hpc/sulis/env.sh`.

GPU interactively (from a Sulis terminal):

```bash
salloc --account=su003-csmm -p gpu-devel -N 1 -n 1 -c 42 --mem-per-cpu=3850 \
       --gres=gpu:ampere_a100:1 --time=01:00:00
srun --pty bash
source ~/msagat/MSAGAT-Net/hpc/sulis/env.sh
nvidia-smi
```

Jupyter on a GPU node (Sulis appnote): inside a `salloc` on `gpu`, load
`JupyterNotebook/7.2.0` after the modules, run
`srun jupyter-notebook --no-browser --ip=$(uname -n)`, then tunnel from the
laptop with
`wsl -d Ubuntu -- ssh -J sulis -N -L 8888:<node>.sulis.hpc:8888 ae6780@<node>.sulis.hpc`.

## 11. Acknowledgement (required in any paper using these runs)

> Calculations were performed using the Sulis Tier 2 HPC platform hosted by the
> Scientific Computing Research Technology Platform at the University of
> Warwick. Sulis is funded by EPSRC Grant EP/T022108/1 and the HPC Midlands+
> consortium.
