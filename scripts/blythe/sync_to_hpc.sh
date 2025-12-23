#!/bin/bash
# Sync conversation-friction to Blythe HPC
# Usage: ./scripts/blythe/sync_to_hpc.sh

set -e

HPC_USER="csuqqj"
HPC_HOST="blythe.scrtp.warwick.ac.uk"
HPC_DIR="u5584851"
PROJECT="conversation-friction"
REMOTE_BASE="/springbrook/share/dcsresearch/${HPC_DIR}"

echo "Syncing to ${HPC_USER}@${HPC_HOST}:${REMOTE_BASE}/${PROJECT}/"

# Sync experiment code
rsync -avz --exclude-from=scripts/blythe/.rsync-exclude \
    experiment/ \
    ${HPC_USER}@${HPC_HOST}:${REMOTE_BASE}/${PROJECT}/experiment/

# Sync slurm scripts
rsync -avz \
    scripts/blythe/slurm/ \
    ${HPC_USER}@${HPC_HOST}:${REMOTE_BASE}/${PROJECT}/slurm/

# Sync pyproject.toml and setup files
rsync -avz \
    pyproject.toml setup.py requirements.txt \
    ${HPC_USER}@${HPC_HOST}:${REMOTE_BASE}/${PROJECT}/

echo "Sync complete!"
echo ""
echo "Next steps on HPC:"
echo "  ssh ${HPC_USER}@${HPC_HOST}"
echo "  cd \$SHARE/${HPC_DIR}/${PROJECT}"
echo "  sbatch slurm/friction_4b.slurm"
