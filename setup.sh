#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URL="https://github.com/yhkimlab/DeepSCF/releases/download/DeepSCF_v1/pretrained-models-v1.tar.gz"

for file in "${ROOT}"/pretrained/{original,r,r_plus_s,r_plus_s_x2,r_plus_s_x3}/{model.pt,data.pkl}; do
    [[ -s "${file}" ]] || { missing=1; break; }
done

if [[ -z "${missing:-}" ]]; then
    echo "Pretrained models are already installed."
else
    curl -fL --retry 3 "${URL}" | tar -xz -C "${ROOT}"
    echo "Pretrained models installed successfully."
fi
