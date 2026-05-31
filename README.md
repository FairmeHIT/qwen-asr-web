# Qwen3-ASR Local Web

简洁的本地音视频转写工具：Qwen3-ASR + FastAPI + 轻量 Web UI。适合在单张 NVIDIA GPU 上转写较长的会议、课程、访谈音视频。

## Features

- Web 页面上传音频/视频并转写
- CLI 批量转写音频/视频
- 自动优先加载本地模型目录
- 支持语种指定、上下文提示、TXT/JSON 输出
- 默认忽略模型、缓存、上传文件和转写结果，适合公开到 GitHub

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
python -m pip install -e .
```

如果你使用本机已备份的 CUDA wheel，可以先安装 GPU 栈：

```bash
python -m pip install --no-index --find-links /mnt/d/codes/torch_install_package \
  torch==2.4.1+cu118 triton==3.0.0 \
  nvidia-cublas-cu11==11.11.3.6 nvidia-cuda-cupti-cu11==11.8.87 \
  nvidia-cuda-nvrtc-cu11==11.8.89 nvidia-cuda-runtime-cu11==11.8.89 \
  nvidia-cudnn-cu11==9.1.0.70 nvidia-cufft-cu11==10.9.0.58 \
  nvidia-curand-cu11==10.3.0.86 nvidia-cusolver-cu11==11.4.1.48 \
  nvidia-cusparse-cu11==11.7.5.86 nvidia-nccl-cu11==2.20.5 \
  nvidia-nvtx-cu11==11.8.86
```

没有本地 wheel 时，请按你的 CUDA/驱动环境安装 PyTorch，再安装本项目依赖。

## Model

默认模型是 `Qwen/Qwen3-ASR-1.7B`。推荐从 ModelScope 下载到本地：

```bash
./run.sh download-model
```

本地模型会按顺序自动查找：

```text
models/Qwen3-ASR-1.7B/config.json
models/config.json
ASR_CHECKPOINT
Qwen/Qwen3-ASR-1.7B
```

也可以手动指定：

```bash
ASR_CHECKPOINT=/path/to/Qwen3-ASR-1.7B ./run.sh web
```

## Usage

启动 Web：

```bash
./run.sh web
```

打开：

```text
http://localhost:8000
```

检查环境：

```bash
./run.sh check
```

命令行转写：

```bash
./run.sh transcribe input.wav -o output.txt --language Chinese
```

安装为 editable 后，也可以直接使用统一 CLI：

```bash
qwen3-asr web
qwen3-asr check
qwen3-asr transcribe input.mp4 -o output.txt --json-output output.json
```

视频文件需要系统安装 `ffmpeg`；纯音频文件不需要。

## Repository Layout

```text
run.sh                  唯一本地脚本入口
requirements.txt        应用依赖
pyproject.toml          包配置与 qwen3-asr CLI
src/qwen_asr_web/
  app.py                FastAPI 应用
  asr.py                模型加载与转写服务
  cli.py                CLI 子命令
  static/               Web 前台
```

## Git Hygiene

`.gitignore` 已排除：

- `.venv/`、`__pycache__/`、构建产物
- `models/`、`.cache/`、大模型权重
- `data/uploads/`、`data/outputs/`
- 常见音视频文件

公开仓库时只提交源码、配置、文档；不要提交模型权重、上传文件、输出文件或本地虚拟环境。
