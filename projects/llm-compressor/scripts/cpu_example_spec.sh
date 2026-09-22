#!/usr/bin/env bash
# Prefetch plan for one cpu-profile example.
# Seven lines: ModelScope model, Hugging Face model, ModelScope dataset,
# Hugging Face dataset, split, dataset config, extra pip packages.
# "-" means that field is unused.

spec_llama3_8b() {
  printf '%s\n' \
    LLM-Research/Meta-Llama-3-8B-Instruct \
    meta-llama/Meta-Llama-3-8B-Instruct
}

spec_no_dataset() {
  printf '%s\n' - - - - -
}

spec_perfectblend() {
  printf '%s\n' \
    mlabonne/open-perfectblend \
    mlabonne/open-perfectblend \
    'train[:512]' \
    - \
    -
}

spec_ultrachat() {
  spec_llama3_8b
  printf '%s\n' \
    HuggingFaceH4/ultrachat_200k \
    HuggingFaceH4/ultrachat_200k \
    "$1" \
    - \
    -
}

cpu_example_spec() {
  case "$1" in
    examples/quantization_w8a8_fp8/llama3_example.py|\
    examples/quantization_non_uniform/quantization_fp8_multiple_strategies.py|\
    examples/quantization_embedding/llama3_example.py|\
    examples/transform/spinquant_example.py)
      spec_llama3_8b
      spec_no_dataset
      ;;
    examples/transform/quip_example.py)
      printf '%s\n' \
        LLM-Research/Meta-Llama-3.1-8B-Instruct \
        meta-llama/Llama-3.1-8B-Instruct
      spec_no_dataset
      ;;
    examples/quantization_w8a8_fp8/qwen3_reranker_example.py)
      printf '%s\n' \
        Qwen/Qwen3-Reranker-8B \
        Qwen/Qwen3-Reranker-8B
      spec_no_dataset
      ;;
    examples/quantization_w8a8_fp8/llava1.5_example.py)
      printf '%s\n' \
        llava-hf/llava-1.5-7b-hf \
        llava-hf/llava-1.5-7b-hf \
        - \
        - \
        - \
        - \
        'torchvision==0.25.0'
      ;;
    examples/quantization_w8a8_fp8/qwen2vl_example.py)
      printf '%s\n' \
        Qwen/Qwen2-VL-7B-Instruct \
        Qwen/Qwen2-VL-7B-Instruct \
        - \
        - \
        - \
        - \
        'torchvision==0.25.0'
      ;;
    examples/quantization_w8a8_fp8/whisper_example.py)
      printf '%s\n' \
        AI-ModelScope/whisper-large-v2 \
        openai/whisper-large-v2 \
        - \
        hf-internal-testing/librispeech_asr_dummy \
        'validation[:1]' \
        clean \
        soundfile
      ;;
    examples/quantization_w4a16/llama3_example.py|\
    examples/quantization_w8a8_int8/llama3_example.py|\
    examples/quantization_w4a8_fp8/llama3_example.py|\
    examples/quantization_attention/llama3_attention.py|\
    examples/quantization_kv_cache/llama3_fp8_kv_example.py|\
    examples/quantization_kv_cache/llama3_fp8_head_kv_example.py|\
    examples/quantization_non_uniform/quantization_int4_int8.py|\
    examples/quantization_non_uniform/quantization_multiple_modifiers.py|\
    examples/awq/llama_example.py|\
    examples/awq/fp8_block_llama_example.py|\
    examples/awq/fp8_dynamic_llama_example.py|\
    examples/awq/w4a8_fp8_llama_example.py)
      spec_llama3_8b
      spec_perfectblend
      ;;
    examples/awq/llama_example_with_masking.py)
      spec_ultrachat 'train_sft[:256]'
      ;;
    examples/custom_dataset_example.py)
      spec_ultrachat 'train_sft[:512]'
      ;;
    examples/quantization_kv_cache/phi3.5_fp8_kv_example.py)
      printf '%s\n' \
        LLM-Research/Phi-3.5-mini-instruct \
        microsoft/Phi-3.5-mini-instruct
      spec_perfectblend
      ;;
    examples/quantization_kv_cache/gemma2_fp8_kv_example.py)
      printf '%s\n' \
        LLM-Research/gemma-2-9b-it \
        google/gemma-2-9b-it
      spec_perfectblend
      ;;
    examples/multimodal_vision/gemma3_example.py)
      printf '%s\n' \
        LLM-Research/gemma-3-4b-it \
        google/gemma-3-4b-it \
        lmms-lab/flickr30k \
        lmms-lab/flickr30k \
        'test[:512]' \
        - \
        'torchvision==0.25.0'
      ;;
    examples/multimodal_vision/qwen2_vl_example.py)
      printf '%s\n' \
        Qwen/Qwen2-VL-2B-Instruct \
        Qwen/Qwen2-VL-2B-Instruct \
        lmms-lab/flickr30k \
        lmms-lab/flickr30k \
        'test[:512]' \
        - \
        'torchvision==0.25.0 qwen-vl-utils'
      ;;
    *)
      return 1
      ;;
  esac
}
