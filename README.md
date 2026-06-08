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
  prompts.jsonl
scripts/
  check_cuda.py
  download_model.py
  infer_transformers.py
  infer_lora.py
  batch_api_infer.py
  train_lora.py
  example_model_patch.py
  patch_resize_fc_example.py
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
- `--system_prompt "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."`
- `--max_new_tokens 128`
- `--temperature 0.7`
- `--top_p 0.9`

推理脚本按 Qwen 官方 quickstart 风格实现：如果 tokenizer 支持 `chat_template`，先用 `apply_chat_template(..., tokenize=False, add_generation_prompt=True)` 渲染 messages，再调用 `tokenizer([text], return_tensors="pt")`，最后使用 `model.generate(**model_inputs)`；否则回退到普通 prompt。

如果命令停在 `[1/5] Loading tokenizer...`，通常说明 `AutoTokenizer.from_pretrained()` 正在联网解析/下载 tokenizer、config 或 chat template。`Qwen/Qwen3-1.7B` 这种写法不会在当前目录创建 `Qwen/` 文件夹；默认缓存位置一般是 `~/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/`。如果想看到一个直观的模型目录，建议先预下载到 `models/Qwen3-1.7B`，再用本地路径推理：

```bash
python scripts/download_model.py \
  --repo_id Qwen/Qwen3-1.7B \
  --local_dir models/Qwen3-1.7B
CUDA_VISIBLE_DEVICES=0 python scripts/infer_transformers.py \
  --model_name_or_path models/Qwen3-1.7B \
  --prompt "请用一句话解释什么是大语言模型。"
```

也可以直接使用 HuggingFace CLI：

```bash
huggingface-cli download Qwen/Qwen3-1.7B --local-dir models/Qwen3-1.7B
```

如果模型已经在缓存中，想避免脚本联网等待，可以加 `--local_files_only`；如果本地没有缓存，它会快速失败并提示你先下载：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_transformers.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --local_files_only \
  --prompt "请用一句话解释什么是大语言模型。"
```

在国内网络环境中，也可以在运行前设置可用的 HuggingFace 镜像端点，例如 `export HF_ENDPOINT=https://hf-mirror.com`，或让服务器管理员配置代理/缓存。建议先执行 `python scripts/download_model.py ...` 确认下载进度，下载完成后再运行推理/训练脚本。



## 调用已部署模型并批量处理 prompt

如果模型已经通过 vLLM、TGI 或其他服务部署成 OpenAI-compatible API，可以用 `scripts/batch_api_infer.py` 批量请求 `/v1/chat/completions`。输入文件是 JSONL，每行一个样本，支持两种格式：

```json
{"id": "p1", "prompt": "请用一句话解释什么是大语言模型。"}
{"id": "p2", "messages": [{"role": "user", "content": "LoRA 适合什么场景？"}]}
```

仓库提供了示例输入 `data/prompts.jsonl`。假设你的服务地址是 `http://127.0.0.1:8000/v1`，模型名是部署时暴露的名字，可以这样批量跑：

```bash
python scripts/batch_api_infer.py \
  --base_url http://127.0.0.1:8000/v1 \
  --model Qwen3-1.7B \
  --input_file data/prompts.jsonl \
  --output_file outputs/batch_api_results.jsonl \
  --concurrency 4 \
  --max_tokens 512 \
  --overwrite
```

输出也是 JSONL，每行包含 `ok`、`response`、`latency_sec`、原始 `raw_response` 或错误信息。若你的服务需要鉴权，可以设置：

```bash
export OPENAI_API_KEY=你的服务token
```

如果是本地 vLLM，一般服务端类似：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model models/Qwen3-1.7B \
  --served-model-name Qwen3-1.7B \
  --host 0.0.0.0 \
  --port 8000
```

## LoRA/QLoRA 训练后如何推理

是的，训练完成后通常需要“基座模型 + adapter”一起加载再推理（不一定必须先离线合并权重）。本仓库提供了 `scripts/infer_lora.py`：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_lora.py \
  --base_model_name_or_path Qwen/Qwen3-1.7B \
  --adapter_path outputs/qwen3_1p7b_lora_test \
  --prompt "请用一句话解释什么是大语言模型。"
```

如果你希望把 LoRA 权重合并回基座模型后再推理（得到纯 Transformers 模型对象），可加：

```bash
--merge_and_unload
```

同样适用于 QLoRA 训练出来的 adapter（前提是 adapter 与 base model 对应）。


## 我想改模型内部结构，不想只做黑盒 SFT

你这个反馈是对的。默认 LoRA/QLoRA 脚本是“先加载基座模型，再注入 LoRA adapter”的标准流程。为了支持你改内部模块，现在训练脚本增加了两个能力：

1. `--preview_modules`：训练前打印 `named_modules()` 预览，方便你定位要改的层名。
2. `--patch_script path/to/your_patch.py`：在注入 LoRA 前执行你自己的 `apply_patch(model)`，可以替换/包装任意子模块。

示例：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_lora_patch_test \
  --preview_modules \
  --patch_script scripts/example_model_patch.py \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4
```

`patch_script` 文件需要提供：

```python
def apply_patch(model):
    # 在这里做模块替换/结构修改
    return model
```

仓库里提供了最小示例：`scripts/example_model_patch.py`。


### 具体示例：改一个全连接层维度，并只训练改动部分

仓库里新增了 `scripts/patch_resize_fc_example.py`，它会：

1. 把 `model.layers.0.mlp.down_proj` 从单层 `nn.Linear` 替换为 `BottleneckFC`（内部隐藏维度改成 1024）；
2. 冻结模型全部参数；
3. 仅放开这个新模块的参数训练。

运行命令（关键是 `--patch_script` + `--disable_lora`）：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_lora.py \
  --model_name_or_path Qwen/Qwen3-1.7B \
  --train_file data/toy_sft.jsonl \
  --output_dir outputs/qwen3_1p7b_custom_fc_only \
  --patch_script scripts/patch_resize_fc_example.py \
  --disable_lora \
  --preview_modules \
  --num_train_epochs 1 \
  --max_length 512 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4
```

如果你想“改模块 + 同时加 LoRA”，去掉 `--disable_lora` 即可。

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
- **推理卡在加载阶段**：如果停在 `[1/5] Loading tokenizer...`，通常是在联网下载 tokenizer/config；`Qwen/Qwen3-1.7B` 默认会进 HuggingFace cache，不会在当前目录生成 `Qwen/`。建议用 `python scripts/download_model.py --repo_id Qwen/Qwen3-1.7B --local_dir models/Qwen3-1.7B` 预下载，再把 `--model_name_or_path` 指向本地目录。
- **`AttributeError: shape` 出现在 `model.generate()`**：这是 tokenizer 返回了 `BatchEncoding`，但旧脚本把它当作 `input_ids` Tensor 传给了 `generate()`。新版脚本已改成 Qwen 官方 quickstart 风格：先渲染 chat template，再用普通 tokenizer 调用生成 `model_inputs`，最后执行 `model.generate(**model_inputs)`，适配 Qwen2.5/Qwen3 等模型。
- **模型下载失败**：确认服务器网络、模型名、HuggingFace 登录状态和本地缓存权限。
- **数据格式不合法**：确认 JSONL 每行都是 JSON object，且包含非空 `text` 或非空 `messages`。
- **bitsandbytes 报错**：优先检查 CUDA 版 PyTorch、bitsandbytes 版本和 GPU 是否可见；QLoRA 通常需要 CUDA GPU。
