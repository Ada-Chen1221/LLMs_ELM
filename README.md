# LLM Lab: Unsloth SFT / GRPO 实验脚手架

这个版本面向“只做 SFT 和 RL，不改模型内部结构”的实验流程：模型加载、LoRA/QLoRA adapter 注入、SFT 和 GRPO RL 训练都交给 **Unsloth + TRL** 完成。保留 Transformers 推理脚本，方便直接验证基座模型或训练后的 adapter。

默认模型是 `Qwen/Qwen3-1.7B`。如果显存或下载条件不足，可以把命令中的 `--model_name_or_path` 切换为 `Qwen/Qwen3-0.6B`；后续也可以切换到 `Qwen/Qwen3-4B`、`Qwen/Qwen3-8B` 或 `Qwen/Qwen3-14B`，同时相应调小 `--max_length`、`--per_device_train_batch_size`、`--num_generations`，并优先使用 `--load_in_4bit`。

## 项目结构

```text
configs/
  infer_qwen3_1p7b.yaml
  train_sft_unsloth_qwen3_1p7b.yaml
  train_rl_grpo_unsloth_qwen3_1p7b.yaml
  train_lora_qwen3_1p7b.yaml          # 旧配置，保留作参数参考
  train_qlora_qwen3_1p7b.yaml         # 旧配置，保留作参数参考
data/
  toy_sft.jsonl
  toy_rl.jsonl
  prompts.jsonl
scripts/
  check_cuda.py
  download_model.py
  infer_transformers.py
  infer_lora.py
  batch_infer_transformers.py
  batch_api_infer.py
  train_sft_unsloth.py
  train_rl_grpo_unsloth.py
  train_lora.py                       # 兼容入口：转到 Unsloth 16-bit LoRA SFT
  train_qlora.py                      # 兼容入口：转到 Unsloth 4-bit QLoRA SFT
src/llm_lab/
  data.py
  model_utils.py
  train_utils.py
  unsloth_utils.py
requirements.txt
README.md
```

> 当前脚本仍以命令行参数为主；`configs/` 中的 YAML 文件用于记录推荐配置，方便后续接入配置加载器。

## 创建环境与安装依赖

建议在服务器上创建独立 conda 环境：

```bash
conda create -n llm-lab python=3.10 -y
conda activate llm-lab
pip install --upgrade pip
pip install -r requirements.txt
```

重要：不要盲目安装最新的 `torch` wheel。`torch` wheel 自带的 CUDA runtime 必须不高于当前 NVIDIA 驱动能支持的 CUDA 版本。当前 `requirements.txt` 默认固定到 `torch==2.5.1+cu121`，更适合 CUDA 12.2 驱动附近的 RTX 3090 服务器。

如果你已经装到了不兼容的 `torch+cu130`，优先这样重装：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_cuda.py
```

如果管理员能升级驱动，也可以保留新版 PyTorch，但需要把 NVIDIA 驱动升级到支持该 PyTorch CUDA build 的版本。脚本不会写入 HuggingFace token；如需访问 gated/private 模型，请在服务器上用 `huggingface-cli login` 登录。

## 检查 CUDA

```bash
python scripts/check_cuda.py
```

该命令会输出 PyTorch 版本、PyTorch CUDA build、`nvidia-smi` 驱动版本、驱动支持的 CUDA Version、CUDA 是否可用、可见 GPU 数量、GPU 名称与显存。脚本不会设置 `CUDA_VISIBLE_DEVICES`，请在命令前自行指定可见 GPU。

## 预下载模型（推荐）

如果网络不稳定，建议先下载模型到本地可见目录，再把训练和推理脚本的模型路径指向本地目录：

```bash
python scripts/download_model.py \
  --repo_id Qwen/Qwen3-1.7B \
  --local_dir models/Qwen3-1.7B
```

在国内网络环境中，也可以在运行前设置可用的 HuggingFace 镜像端点，例如 `export HF_ENDPOINT=https://hf-mirror.com`，或让服务器管理员配置代理/缓存。

## Unsloth SFT（LoRA / QLoRA）

推荐入口：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_sft_unsloth.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_unsloth_sft \
  --max_length 1024 \
  --load_in_4bit
```

关键参数：

- `--load_in_4bit` / `--no-load_in_4bit`：开启时是 QLoRA 风格，关闭时是 16-bit LoRA 风格。
- `--lora_r`、`--lora_alpha`、`--lora_dropout`、`--target_modules`：LoRA adapter 参数。默认 target modules 适配 Qwen/LLaMA 类 causal LM：`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。
- `--gradient_checkpointing`：默认开启，并使用 Unsloth 的 checkpointing 路径以降低显存。
- `--dtype auto|float16|bfloat16|float32`：`auto` 会交给 Unsloth/硬件自动选择。

训练结束后会把 adapter 和 tokenizer 保存到 `--output_dir`。该流程不手动改模型内部实现，只通过 Unsloth 的 `FastLanguageModel.from_pretrained()` 和 `FastLanguageModel.get_peft_model()` 加载模型并挂载 adapter。

## Unsloth GRPO RL

最小 GRPO 入口：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_rl_grpo_unsloth.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_rl.jsonl \
  --output_dir outputs/qwen3_1p7b_unsloth_grpo \
  --max_steps 10 \
  --num_generations 2 \
  --load_in_4bit
```

`data/toy_rl.jsonl` 每行一个 JSON object，支持：

```json
{"prompt": "请只输出 2+2 的阿拉伯数字答案。", "answer": "4"}
{"messages": [{"role": "user", "content": "请只输出 3*3 的阿拉伯数字答案。"}], "answer": "9"}
```

脚本内置两个示例 reward：

- `exact_answer_reward`：如果数据中有 `answer` / `target` / `reference` 字段，则对完全匹配或包含答案的 completion 给分。
- `format_reward`：给非空、较简洁且没有明显重复的 completion 一个小奖励。

真实实验中建议把 `scripts/train_rl_grpo_unsloth.py` 里的 reward function 替换成任务相关的规则、判别器或 judge 模型评分。

## 推理

### 基座模型推理

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --prompt "请用一句话解释什么是大语言模型。"
```

如果模型已经在缓存中，想避免脚本联网等待，可以加 `--local_files_only`；如果本地没有缓存，它会快速失败并提示你先下载。

### Adapter 推理

SFT 或 GRPO 训练完成后，通常需要“基座模型 + adapter”一起加载再推理：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_lora.py \
  --base_model_name_or_path Qwen/Qwen3-1.7B \
  --adapter_path outputs/qwen3_1p7b_unsloth_sft \
  --prompt "请用一句话解释 LoRA 微调。"
```

如果你希望把 LoRA 权重合并回基座模型后再推理，可加：

```bash
--merge_and_unload
```

## 批量推理

输入文件每行一个 JSON object，支持两种格式：

```json
{"id": "p1", "prompt": "请用一句话解释什么是大语言模型。"}
{"id": "p2", "messages": [{"role": "user", "content": "LoRA 适合什么场景？"}]}
```

运行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_infer_transformers.py \
  --model_name_or_path models/Qwen3-1.7B \
  --input_file data/prompts.jsonl \
  --output_file outputs/predictions.jsonl \
  --batch_size 2
```

## 多卡示例

SFT：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 scripts/train_sft_unsloth.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_unsloth_sft_ddp
```

GRPO：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 scripts/train_rl_grpo_unsloth.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_rl.jsonl \
  --output_dir outputs/qwen3_1p7b_unsloth_grpo_ddp \
  --num_generations 2
```

多卡训练前建议先单卡小数据跑通。GRPO 的显存随 `num_generations`、prompt/completion 长度和 batch 增长很明显，先用小步数确认 reward 和数据格式正确。

## SFT 数据格式

`data/toy_sft.jsonl` 每行一个 JSON object，支持：

```json
{"text": "用户：...\n助手：..."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

如果 tokenizer 支持 chat template，`messages` 会通过 `tokenizer.apply_chat_template(..., tokenize=False)` 转成训练文本；否则退化为简单的 `role: content` 文本。

## 常见问题

- **加载阶段很慢**：通常是在联网下载 tokenizer/config/model；建议先用 `scripts/download_model.py` 预下载，再使用本地路径。
- **CUDA 不可用**：先运行 `python scripts/check_cuda.py`，确认 PyTorch CUDA build、驱动和可见 GPU 是否匹配。
- **Unsloth / bitsandbytes 报错**：优先检查 CUDA 版 PyTorch、bitsandbytes、unsloth 与 GPU 架构是否兼容；QLoRA 通常需要 CUDA GPU。
- **GRPO 无学习信号**：检查 reward function 是否真的给出了非零分数；先用 `data/toy_rl.jsonl` 和很小的 `--max_steps` 跑通，再换真实数据。
