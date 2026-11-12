#!/usr/bin/env bash
# Rebuilds every model and every number in the README from scratch (CPU only, ~12 min on 2 cores).
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p data/external/goemotions
GE=https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data
for f in train dev test; do
  [ -f data/external/goemotions/$f.tsv ] || curl -sL -o data/external/goemotions/$f.tsv $GE/$f.tsv
done
[ -f data/external/goemotions/emotions.txt ] || curl -sL -o data/external/goemotions/emotions.txt $GE/emotions.txt

if [ ! -f models/glove_50k_100d.npz ]; then
  curl -sL -o /tmp/glove100.gz https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz
  python scripts/build_embeddings.py /tmp/glove100.gz
fi

python scripts/audit_teacher.py              # 1. audit the Llama judge outputs  -> reports/teacher_audit.json
python scripts/train_emotion.py              # 2. human-label emotion encoder    -> reports/emotion_metrics.json
python scripts/validate_teacher_emotions.py  # 3. Llama judge vs emotion model   -> reports/teacher_emotion_validity.json
python scripts/train_lite.py                 # 4. distil + cross-validate        -> reports/lite_metrics.json
python scripts/validate_against_humans.py --n 1000   # 5. lite judge vs human labels -> reports/human_validation.json
python scripts/make_figures.py
python -m pytest -q
