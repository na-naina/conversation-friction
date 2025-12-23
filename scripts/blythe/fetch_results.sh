#!/bin/bash
# Fetch results from Blythe HPC
# Usage: ./scripts/blythe/fetch_results.sh

set -e

HPC_USER="csuqqj"
HPC_HOST="blythe.scrtp.warwick.ac.uk"
HPC_DIR="u5584851"
PROJECT="conversation-friction"
REMOTE_BASE="/springbrook/share/dcsresearch/${HPC_DIR}"

echo "Fetching results from ${HPC_USER}@${HPC_HOST}..."

# Fetch results
rsync -avz \
    ${HPC_USER}@${HPC_HOST}:${REMOTE_BASE}/${PROJECT}/data/results/ \
    data/results/

# Fetch logs
rsync -avz \
    ${HPC_USER}@${HPC_HOST}:${REMOTE_BASE}/${PROJECT}/logs/ \
    logs/

echo "Results fetched to data/results/"
echo "Logs fetched to logs/"
