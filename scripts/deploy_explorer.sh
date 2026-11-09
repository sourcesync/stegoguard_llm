#!/usr/bin/env bash
# Build the results explorer from the local data and deploy it to Cloud Run.
#
# Needs: the data bundle unpacked under outputs/pilot_v0, python3, and a
# logged-in gcloud (`gcloud auth login`, `gcloud config set project ...`).
# The service requires authentication; grant access with:
#   gcloud run services add-iam-policy-binding stegoguard-explorer \
#       --region us-west1 --member user:someone@example.com --role roles/run.invoker
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
    --no-allow-unauthenticated \
    --memory 256Mi \
    --max-instances 2

echo
echo "Deployed. Open it through an authenticated proxy:"
echo "  gcloud run services proxy $SERVICE --region $REGION --project $PROJECT"
