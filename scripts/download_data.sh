#!/usr/bin/env bash
# Download every proposal dataset into data/ (git-ignored). Safe to re-run.
# Needs: Kaggle login (`kaggle auth login`), git, curl, the linkedin-assistant conda env.
# Run from the repo root:  bash scripts/download_data.sh
set -euo pipefail

ENV_BIN=/opt/anaconda3/envs/linkedin-assistant/bin
KAGGLE="$ENV_BIN/kaggle"
RAW=data/raw
EXT=data/external
mkdir -p "$RAW" "$EXT"

# --- Jobs ---
"$KAGGLE" datasets download arshkon/linkedin-job-postings -p "$RAW/linkedin_job_postings" --unzip
"$KAGGLE" datasets download sindhumadhurii/linkedin-job-posts-insights-dataset -p "$RAW/linkedin_job_posts_insights" --unzip
"$ENV_BIN/python" -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='datastax/linkedin_job_listings', repo_type='dataset', local_dir='$RAW/hf_datastax_linkedin')"

# --- Resumes ---
"$KAGGLE" datasets download snehaanbhawal/resume-dataset -p "$RAW/resume_dataset" --unzip
if [ ! -d "$RAW/resume_corpus" ]; then
  git clone -q --depth 1 https://github.com/carbscountry/resume_corpus.git "$RAW/resume_corpus"
  rm -rf "$RAW/resume_corpus/.git"   # keep plain data, not a nested repo
fi
unzip -q -o "$RAW/resume_corpus/resumes_corpus.zip" -d "$RAW/resume_corpus/extracted"
unzip -q -o "$RAW/resume_corpus/resume_samples.zip" -d "$RAW/resume_corpus/extracted_samples"

# --- Taxonomy (D14: O*NET replaces Lightcast) ---
ONET_ZIP=db_31_0_csv.zip
mkdir -p "$EXT/onet"
curl -sL -A "Mozilla/5.0" "https://www.onetcenter.org/dl_files/database/$ONET_ZIP" -o "$EXT/onet/$ONET_ZIP"
unzip -q -o "$EXT/onet/$ONET_ZIP" -d "$EXT/onet" && rm "$EXT/onet/$ONET_ZIP"

echo "Done."
du -sh "$RAW"/* "$EXT"/*
