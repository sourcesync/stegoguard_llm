#!/usr/bin/env bash
# Build the results explorer from the local data and deploy it to Cloud Run.
#
# Needs: the data bundle unpacked under outputs/pilot_v0, python3, and a
# logged-in gcloud (`gcloud auth login`, `gcloud config set project ...`).
# The service is public: anyone with its https URL can open it. If your
# organization blocks public Cloud Run services, the deploy succeeds but the
# URL returns 403 (see the main README).
#
# Options (environment variables): SERVICE, REGION, PROJECT.
# Added after the experiments; not covered by the freeze manifests.
set -euo pipefail

SERVICE="${SERVICE:-stegoguard-explorer}"
REGION="${REGION:-us-west1}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
[ -n "$PROJECT" ] || { echo "No GCP project: set PROJECT or run 'gcloud config set project ...'" >&2; exit 1; }

cd "$(dirname "$0")/.."

python3 scripts/export_explorer_bundle.py
python3 scripts/build_explorer_page.py

# Stage outside the repository so gcloud does not apply .gitignore to the data.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp app/main.py app/requirements.txt app/Dockerfile "$STAGE/"
mkdir "$STAGE/data"
cp outputs/explorer/index.html outputs/explorer/explorer_bundle.json "$STAGE/data/"

gcloud run deploy "$SERVICE" \
    --source "$STAGE" \
    --project "$PROJECT" \
    --region "$REGION" \
    --allow-unauthenticated \
    --memory 256Mi \
    --max-instances 2

echo
echo "Deployed. Public URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --project "$PROJECT" --format 'value(status.url)'
