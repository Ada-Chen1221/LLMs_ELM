# LLM Lab: Transformers 推理与 LoRA/QLoRA 训练脚手架

这是一个最小可运行的 HuggingFace Transformers 原生脚手架，用于在实验室 RTX 3090 环境中先跑通：

- Transformers 原生 causal LM 推理
- LoRA SFT 训练
- QLoRA 4-bit SFT 训练

默认模型是 `Qwen/Qwen3-1.7B`。如果显存或下载条件不足，可以把命令中的 `--model_name_or_path` 切换为 `Qwen/Qwen3-0.6B`；后续也可以切换到 `Qwen/Qwen3-4B`、`Qwen/Qwen3-8B` 或 `Qwen/Qwen3-14B`，同时相应调小 `--max_length`、`--per_device_train_batch_size`，或使用 QLoRA。

## 项目结构

```text
configs/
  infer_qwen3_1p7b.yaml
  train_lora_qwen3_1p7b.yaml
  train_qlora_qwen3_1p7b.yaml
data/
  toy_sft.jsonl
scripts/
  check_cuda.py
  infer_transformers.py
  train_lora.py
  train_qlora.py
src/llm_lab/
  __init__.py
  data.py
  model_utils.py
  train_utils.py
requirements.txt
README.md
```

> 当前脚本以命令行参数为主；`configs/` 中的 YAML 文件用于记录推荐配置，方便后续接入配置加载器。

## 创建环境与安装依赖

建议在服务器上创建独立 conda 环境：

```bash
conda create -n llm-lab python=3.10 -y
conda activate llm-lab
pip install --upgrade pip
pip install -r requirements.txt
```

重要：不要盲目安装最新的 `torch` wheel。`torch` wheel 自带的 CUDA runtime 必须不高于当前 NVIDIA 驱动能支持的 CUDA 版本。比如你看到的日志里 `PyTorch version: 2.12.0+cu130` / `PyTorch CUDA build: 13.0`，但驱动只报告 `found version 12020`（约等于驱动支持 CUDA 12.2），因此 PyTorch 会判定 CUDA 不可用。当前 `requirements.txt` 默认固定到 `torch==2.5.1+cu121`，更适合这类 CUDA 12.2 驱动的 RTX 3090 服务器。

如果你已经装到了不兼容的 `torch+cu130`，优先这样重装：

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/check_cuda.py
```

如果管理员能升级驱动，也可以保留新版 PyTorch，但需要把 NVIDIA 驱动升级到支持该 PyTorch CUDA build 的版本。CUDA 13.0 需要 580 系列或更新的 Linux 驱动；CUDA 12.2 驱动无法运行 `+cu130` 的 PyTorch。

如果 `bitsandbytes` 与 CUDA/PyTorch 版本不匹配，请根据服务器 CUDA 版本重新安装兼容的 PyTorch 和 bitsandbytes。脚本不会写入 HuggingFace token；如需访问 gated/private 模型，请在服务器上用 `huggingface-cli login` 登录。

## 检查 CUDA

```bash
python scripts/check_cuda.py
```

该命令会输出 PyTorch 版本、PyTorch CUDA build、`nvidia-smi` 驱动版本、驱动支持的 CUDA Version、CUDA 是否可用、可见 GPU 数量、GPU 名称与显存。脚本不会设置 `CUDA_VISIBLE_DEVICES`，请在命令前自行指定可见 GPU。若 PyTorch CUDA build 高于驱动支持版本，脚本会给出重装 cu121 PyTorch 或升级驱动的提示。

## 单卡 Transformers 推理

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --prompt "请用一句话解释什么是大语言模型。"
```

常用参数：

- `--dtype auto|float16|bfloat16|float32`
- `--device_map auto`
- `--max_new_tokens 128`
- `--temperature 0.7`
- `--top_p 0.9`

如果 tokenizer 支持 `chat_template`，脚本会优先使用 `apply_chat_template`；否则回退到普通 prompt。

## 单卡 LoRA 训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_lora_test \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

LoRA 默认 target modules 适配 Qwen/LLaMA 类 causal LM：

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

训练结束后会把 adapter 和 tokenizer 保存到 `--output_dir`。

## 单卡 QLoRA 训练

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_qlora_test \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

QLoRA 默认使用：

- 4-bit `BitsAndBytesConfig`
- `nf4`
- double quant
- compute dtype `float16`

RTX 3090 上默认使用 fp16 更稳；如确有需要可增加 `--bf16`，脚本会关闭 fp16。

## 双卡训练示例

LoRA 双卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 scripts/train_lora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_lora_2gpu \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

QLoRA 双卡：

```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 scripts/train_qlora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_qlora_2gpu \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

如果只想让脚本看到第 2、3 张物理卡，可以使用：

```bash
CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes=2 scripts/train_lora.py ...
```

## 切换模型

只需要修改命令中的 `--model_name_or_path`：

```bash
--model_name_or_path Qwen/Qwen3-0.6B
--model_name_or_path Qwen/Qwen3-4B
--model_name_or_path Qwen/Qwen3-8B
--model_name_or_path Qwen/Qwen3-14B
```

建议：

- 0.6B/1.7B：优先用于 smoke test。
- 4B/8B：单卡 24GB 可优先尝试 QLoRA，LoRA 需要更谨慎调小 batch/length。
- 14B：建议 QLoRA + 多卡，先从 `--max_length 512` 和 batch size 1 开始。

## 数据格式

`data/toy_sft.jsonl` 支持两种格式：

```json
{"text": "用户：...\n助手：..."}
```

或：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

如果 tokenizer 支持 chat template，`messages` 会通过 `tokenizer.apply_chat_template(..., tokenize=False)` 转成训练文本；否则退化为简单的 `role: content` 文本。

## 必跑验收命令

```bash
python scripts/check_cuda.py
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --prompt "请用一句话解释什么是大语言模型。"
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_lora_test \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_qlora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_qlora_test \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```

## 常见问题

- **CUDA 不可用**：运行 `python scripts/check_cuda.py`，优先确认 PyTorch CUDA build 是否高于 `nvidia-smi` 显示的 CUDA Version。若出现 `torch+cu130` 但驱动只支持 CUDA 12.2，请重装 `torch==2.5.1+cu121` 或升级 NVIDIA 驱动。
- **模型下载失败**：确认服务器网络、模型名、HuggingFace 登录状态和本地缓存权限。
- **数据格式不合法**：确认 JSONL 每行都是 JSON object，且包含非空 `text` 或非空 `messages`。
- **bitsandbytes 报错**：优先检查 CUDA 版 PyTorch、bitsandbytes 版本和 GPU 是否可见；QLoRA 通常需要 CUDA GPU。
