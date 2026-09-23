#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <profile>" >&2
  exit 2
fi

PROFILE="$1"
# A profile selects the smallest dependency set needed by one example family;
# it is not a transformers feature flag. Keep unknown profiles failing before
# any package is installed so a manifest typo cannot run a partial setup.
case "$PROFILE" in
  generation)
    DEPS=(accelerate)
    ;;
  glue)
    DEPS=(accelerate datasets evaluate scikit-learn)
    ;;
  small-training)
    DEPS=(accelerate datasets evaluate seqeval sentencepiece tiktoken)
    ;;
  lm)
    DEPS=(accelerate datasets evaluate)
    ;;
  seq2seq)
    DEPS=(accelerate datasets evaluate sacrebleu rouge-score nltk sentencepiece tiktoken)
    ;;
  *)
    echo "unknown profile: $PROFILE (supported: generation glue small-training lm seq2seq)" >&2
    exit 1
    ;;
esac

: "${TARGET_ROOT:?TARGET_ROOT is required}"
: "${GITHUB_ENV:?GITHUB_ENV is required}"
export PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
export PIP_TRUSTED_HOST="repo.huaweicloud.com"
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python -m pip install -U pip
# The Ascend image normally contains a compatible torch/torch_npu pair. Reuse
# it when possible because these wheels are large; install only on a missing
# or unusable image stack.
if python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.device_count())"; then
  echo "reusing image torch stack"
else
  python -m pip install --extra-index-url https://repo.huaweicloud.com/ascend/repos/pypi \
    torch==2.9.0 torch_npu==2.9.0.post2
fi
# Install the current target checkout. Its dependencies (regex, tokenizers,
# huggingface-hub, safetensors, ...) are resolved by pip from the index; this is
# safe for the NPU torch stack because transformers does not depend on torch_npu
# and its torch requirement is already satisfied by the image build.
python -m pip install -e "$TARGET_ROOT"
python -m pip install "${DEPS[@]}"

# Pre-download example model weights from ModelScope (China-reachable) so the
# examples load them from a local path instead of the blocked HuggingFace CDN.
# The returned local snapshot dirs are exported as env vars for the example
# (overlay_args in examples_manifest.yaml reference them via ${VAR}).
# Pinned: modelscope>=1.38 splits the hub code into modelscope-hub, and the
# fresh 1.40.1 wheel's loose ">=0.4.2" floor breaks import when the mirror
# lags on hub 0.4.3. 1.37.0 is the last pre-split line.
python -m pip install "modelscope==1.37.0"
python - <<'PY'
import os
from modelscope import snapshot_download

MODEL_CACHE = os.environ.get("MODELSCOPE_CACHE", os.path.expanduser("~/.cache/modelscope"))
mapping = {
    "DISTILBERT_PATH": "distilbert/distilbert-base-uncased",
    "TINYGPT2_PATH": "sshleifer/tiny-gpt2",
    "TINYMBART_PATH": "sshleifer/tiny-mbart",
}
for env_name, model_id in mapping.items():
    local = snapshot_download(model_id, cache_dir=MODEL_CACHE)
    if model_id == "sshleifer/tiny-mbart":
        import json

        cfg_path = os.path.join(local, "config.json")
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        # The ModelScope snapshot lacks decoder_start_token_id; transformers
        # main no longer falls back to model-class defaults, so the mbart
        # summarize/seq2seq_qa examples raise "Make sure that
        # `config.decoder_start_token_id` is correctly defined". Inject the
        # mbart standard value (eos </s> id) idempotently.
        if cfg.get("decoder_start_token_id") is None:
            cfg["decoder_start_token_id"] = 2
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=2)
    with open(os.environ["GITHUB_ENV"], "a") as fh:
        fh.write(f"{env_name}={local}\n")
PY

# tiny xlnet exists only on HuggingFace (ModelScope 404s the repo); pull it via
# the China-reachable hf-mirror.com mirror for the beam-search QA examples.
# Kept non-fatal so a mirror outage cannot fail the unrelated example jobs.
if TINYXLNET=$(python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    "sshleifer/tiny-xlnet-base-cased",
    endpoint="https://hf-mirror.com",
))
PY
); then
  echo "TINYXLNET_PATH=$TINYXLNET" >> "$GITHUB_ENV"
else
  echo "warning: tiny xlnet download via hf-mirror.com failed; beam-search examples will fail" >&2
fi

# LM-family examples infer the dataset loader from the train_file suffix and
# reject the extensionless wiki_text/wiki_00 fixture, so expose it as train.txt
# in the job output dir for the ${CI_OUTPUT_DIR}/train.txt overlay args.
# sentencepiece note: transformers main loads sentencepiece-only vocab files
# (xlnet spiece.model, mbart sentencepiece.bpe.model) via SentencePieceExtractor
# and only falls back to a tiktoken parse when that import/extraction fails, so
# the sentencepiece package must be installed or loading crashes with
# "ValueError: Error parsing line ... in spiece.model".
: "${CI_OUTPUT_DIR:=$GITHUB_WORKSPACE/output}"
mkdir -p "$CI_OUTPUT_DIR"
cp "$TARGET_ROOT/tests/fixtures/tests_samples/wiki_text/wiki_00" "$CI_OUTPUT_DIR/train.txt"

# Trainer-based classification examples map string labels through label_to_id
# inside datasets.map(), but datasets keeps the original string column type and
# casts the ints back to str, so torch_default_data_collator crashes with
# "too many dimensions 'str'". Emit int-label copies of the MRPC fixture for
# the run_glue.py / run_classification.py overlay args.
python - "$TARGET_ROOT" "$CI_OUTPUT_DIR" <<'PY'
import csv
import os
import sys

target, out = sys.argv[1], sys.argv[2]
label2id = {"equivalent": "0", "not_equivalent": "1"}
for split in ("train", "dev"):
    src = os.path.join(target, "tests", "fixtures", "tests_samples", "MRPC", f"{split}.csv")
    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    with open(os.path.join(out, f"mrpc_{split}.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row["label"] = label2id[row["label"]]
            writer.writerow(row)
PY
