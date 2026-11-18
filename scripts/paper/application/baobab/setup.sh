#!/usr/bin/env bash
# Baobab, on the GPU node of an interactive allocation (`ssh <node>` lands in
# it): the code, the Python env and boostCopula's R library on the node's own
# disk, /scratch/$USER-npcc. Importing torch from BeeGFS scratch is
# metadata-bound and crawls (6 MB read in 7 minutes on 2026-11-18), and the
# login node's R build crawled too; the node's disk does both in minutes, but
# may not outlive the allocation, so a new allocation runs this again.
#
# Before: the code as a git bundle (`git bundle create npcc.bundle <branch>`;
# the repository is private and the cluster has no GitHub credentials), the
# local `uv` binary in ~/scratch/npcc-app/bin/, and the data, with their
# SHA-256, in ~/scratch/npcc-data/meteoswiss_zanetta2023/. The same lock and
# extras as every run of this repository.
set -euo pipefail
B=${B:-$HOME/scratch}
A=$B/npcc-app
N=${N:-/scratch/$USER-npcc}
mkdir -p "$N"
[ -e "$HOME/npcc-data" ] || ln -s "$B/npcc-data" "$HOME/npcc-data"
export UV_CACHE_DIR=$N/uv-cache UV_PYTHON_INSTALL_DIR=$N/python
echo "== $(date '+%F %T') code on $(hostname)"
[ -d "$N/npcc" ] || git clone -q "$A/npcc.bundle" "$N/npcc"
cd "$N/npcc"
git fetch -q "$A/npcc.bundle" 'refs/heads/*:refs/remotes/bundle/*'
echo "== $(date '+%F %T') python env"
"$A/bin/uv" sync --frozen --python 3.12 --managed-python --extra cu130 --extra backends --extra experiments 2>&1 | tail -3
"$A/bin/uv" pip freeze --python "$N/npcc/.venv/bin/python" > "$N/freeze.txt"
"$N/npcc/.venv/bin/python" -c "import torch, tabpfn, tabicl, npcc; print(torch.__version__, torch.cuda.get_device_name(0))"
echo "== $(date '+%F %T') R"
module load GCC/15.2.0 CMake/4.2.1 R/4.6.1
export R_LIBS_USER=$N/R/4.6
mkdir -p "$R_LIBS_USER"
Rscript -e 'install.packages(c("RcppEigen","colorspace","RcppParallel","Rfast","VineCopula"), lib=Sys.getenv("R_LIBS_USER"), repos="https://cloud.r-project.org", Ncpus=4, quiet=TRUE)'
# Its dependencies are installed above: remotes would otherwise try the
# module's read-only library for them.
Rscript -e "remotes::install_github('jobstdavid/boostCopula@746b19152ab71517c4e0dce6d651c9c43705efe9', lib='$R_LIBS_USER', upgrade='never', dependencies=FALSE, quiet=TRUE)"
Rscript -e 'for (p in c("boostCopula","VineCopula","Rfast","RcppParallel","RcppEigen","Rcpp")) cat(p, as.character(packageVersion(p)), "\n"); cat(R.version.string, "\n")' | tee "$N/r_versions.txt"
echo "== $(date '+%F %T') done"
