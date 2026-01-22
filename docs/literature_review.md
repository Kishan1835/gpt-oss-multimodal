# Literature Review

## Vision–Language Models
Recent advances in multimodal learning have enabled large language models to process and reason over visual inputs by aligning visual representations with textual embeddings. Vision–Language Models (VLMs) combine pretrained vision encoders with large language models using lightweight alignment mechanisms.

## LLaVA
LLaVA (Large Language and Vision Assistant) introduces a simple yet effective projection-based alignment between a frozen vision encoder and a large language model. By training only a small projection layer on multimodal instruction data, LLaVA demonstrates strong performance on image captioning and visual question answering tasks.

## BLIP-2
BLIP-2 proposes a Query Transformer (Q-Former) to bridge vision encoders and language models. By keeping both the vision encoder and language model frozen, BLIP-2 achieves efficient multimodal alignment with reduced training cost.

## MiniGPT-4
MiniGPT-4 aligns a vision encoder with a large language model using a projection layer trained on image–text pairs. It demonstrates that strong multimodal reasoning can be achieved without full model fine-tuning.

## Gap Identification
While existing approaches successfully integrate vision and language, most models rely on closed or restricted-weight language models. There is limited exploration of extending open-weight large language models, such as GPT-OSS, with multimodal vision capabilities while preserving openness and reproducibility.
