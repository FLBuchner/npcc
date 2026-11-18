# Baobab: sourced by launch_full.sh (SETUP) on the GPU node. R for boostCopula
# from the node's own library (setup_node.sh); matplotlib's cache there too.
N=${N:-/scratch/$USER-npcc}
if ! type module > /dev/null 2>&1; then
  for init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do
    [ -r "$init" ] && source "$init" && break
  done
fi
module load GCC/15.2.0 R/4.6.1
export R_LIBS_USER=$N/R/4.6
export MPLCONFIGDIR=$N/mpl
# TabICL's weights are in the Hugging Face cache; a job never reaches the hub.
export HF_HUB_OFFLINE=1
