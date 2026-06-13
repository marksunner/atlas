# Step 3.7 Flash (NVFP4) on DGX Spark — Quickstart (2-node EP=2)

Step 3.7 Flash is a 45-layer, 288-expert MoE. The NVFP4 weights are ~140 GB total,
so this model requires at least two DGX Sparks in expert-parallel (EP=2, TP=1).

## Prerequisites

- 2× NVIDIA DGX Spark (GB10, 128 GB unified memory each)
- Docker with NVIDIA Container Toolkit
- RoCE / high-speed interconnect between nodes (NCCL transport)
- Model weights downloaded on both nodes

## 1. Download model weights

On **both** Spark nodes:

```bash
huggingface-cli download stepfun-ai/Step-3.7-Flash-NVFP4 \
  --local-dir /models/Step-3.7-Flash-NVFP4
```

This downloads ~140 GB of weights as real files (no symlinks).

## 2. Build the Docker image

From the repository root (this branch):

```bash
docker build -f docker/gb10/Dockerfile -t atlas-step3p7:latest .
```

Build takes ~10-15 minutes (Rust compilation). Make the image available on both
nodes — build on each, or:

```bash
docker save atlas-step3p7:latest | ssh <other-node> docker load
```

## 3. Launch EP=2 (two nodes)

**Important memory note:** Step 3.7 has 45 attention layers (12 full + 33 sliding
window), all backed by KV cache. With NVFP4 weights + the MoE transpose pass, memory
is tight on 128 GB Sparks. Use conservative settings:

- `--max-seq-len 512` (sufficient for testing; higher values may OOM)
- `--max-prefill-tokens 512` (reduces buffer arena size)
- `--max-batch-size 1` (EP v1 enforces this anyway)
- `--kv-cache-dtype fp8` (default, saves KV memory)
- `--kv-high-precision-layers 0` (saves KV memory; use `auto` if you have headroom)
- `--gpu-memory-utilization 0.95` (allows tighter packing)

### Node 1 (rank 0 — head, serves HTTP):

```bash
docker run -d --name atlas-step3p7-rank0 \
  --gpus all --ipc=host --network=host \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e ATLAS_SKIP_OOM_PREFLIGHT=1 \
  -v /models/Step-3.7-Flash-NVFP4:/model \
  atlas-step3p7:latest \
  serve /model \
  --world-size 2 --ep-size 2 --rank 0 \
  --master-addr <NODE1_CLUSTER_IP> \
  --max-seq-len 512 \
  --max-prefill-tokens 512 \
  --max-batch-size 1 \
  --kv-cache-dtype fp8 \
  --kv-high-precision-layers 0 \
  --gpu-memory-utilization 0.95 \
  --bind 0.0.0.0
```

### Node 2 (rank 1 — EP worker):

```bash
docker run -d --name atlas-step3p7-rank1 \
  --gpus all --ipc=host --network=host \
  -e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0 \
  -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e ATLAS_SKIP_OOM_PREFLIGHT=1 \
  -v /models/Step-3.7-Flash-NVFP4:/model \
  atlas-step3p7:latest \
  serve /model \
  --world-size 2 --ep-size 2 --rank 1 \
  --master-addr <NODE1_CLUSTER_IP> \
  --max-seq-len 512 \
  --max-prefill-tokens 512 \
  --max-batch-size 1 \
  --kv-cache-dtype fp8 \
  --kv-high-precision-layers 0 \
  --gpu-memory-utilization 0.95
```

Replace `<NODE1_CLUSTER_IP>` with the IP of Node 1 on the high-speed interconnect
(e.g., `10.0.0.4` for DGX Spark RoCE network).

**NCCL environment notes:** The `NCCL_IB_HCA` and `NCCL_SOCKET_IFNAME` values above
are for the DGX Spark's ConnectX-7 RoCE interfaces. If your network topology differs,
adjust these to match your RDMA/InfiniBand device names.

**Model path note:** If you downloaded via `huggingface-cli download` without
`--local-dir`, the weights live in the HF cache. In that case, mount the cache and
use the HF model ID:

```bash
-v ~/.cache/huggingface:/root/.cache/huggingface \
atlas-step3p7:latest serve stepfun-ai/Step-3.7-Flash-NVFP4 ...
```

## 4. Verify

Wait ~90 seconds for model loading and NCCL connection, then:

```bash
# Check logs — look for "Listening on 0.0.0.0:8888"
docker logs atlas-step3p7-rank0

# Basic coherence test
curl http://<NODE1_IP>:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step3p7",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Expected: coherent response ("2+2 is 4" or similar), ~18 tok/s decode.

### With thinking enabled (default):

Omit `chat_template_kwargs` — thinking is on by default:

```bash
curl http://<NODE1_IP>:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "step3p7",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 200,
    "temperature": 0
  }'
```

## Known issues

- **Long generation (>100 tokens) can loop.** Use `repetition_penalty: 1.1` or similar
  for longer outputs. This reproduces on pre-patch Atlas as well — not specific to this
  branch.
- **Memory is tight.** `--max-seq-len 512` works reliably; higher values may cause OOM
  and system unresponsiveness. If the Spark becomes unresponsive, power-cycle and use
  more conservative settings.
- **First request may be slow.** CUDA kernel compilation on the first inference pass can
  take several seconds.

## What you should see in the logs

Healthy startup shows:
```
step3p7: built 45 layers (45 attention)     ← all layers loaded
Weights: 70.53 GB                           ← per-rank weight size
Listening on 0.0.0.0:8888                   ← ready to serve
Swap space: 3 GB at /tmp/atlas-swap/        ← swap configured
```
