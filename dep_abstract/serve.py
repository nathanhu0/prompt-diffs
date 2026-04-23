"""Launch one or more vLLM servers and wait for them to be healthy.

Usage:
    # Single model
    python serve.py --model meta-llama/Llama-3.1-8B-Instruct --gpu 0 --port 8001

    # Multiple models (one per GPU)
    python serve.py \
        --model meta-llama/Llama-3.1-8B-Instruct --gpu 0 --port 8001 \
        --model Qwen/Qwen2.5-7B-Instruct --gpu 1 --port 8002

    # Then run your experiment against http://localhost:8001/v1 etc.
    # Ctrl-C to shut down all servers.
"""
import argparse
import atexit
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def find_free_port(low=10000, high=60000):
    """Find a random free port in the given range."""
    for _ in range(100):
        port = random.randint(low, high)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Could not find a free port")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", required=True,
                        help="Model ID (repeat for multiple)")
    parser.add_argument("--gpu", action="append", type=int, required=True,
                        help="GPU index for each model (repeat for multiple)")
    parser.add_argument("--port", action="append", type=int, required=True,
                        help="Port for each model (repeat for multiple)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--timeout", type=int, default=300,
                        help="Max seconds to wait for each server to be healthy")
    args = parser.parse_args()

    if not (len(args.model) == len(args.gpu) == len(args.port)):
        parser.error("Must provide equal number of --model, --gpu, and --port")
    return args


def launch_server(model, gpu, port, gpu_mem_util, max_model_len, data_parallel=1):
    """Start a vLLM server as a subprocess, return the Popen handle.

    gpu: single GPU index (data_parallel=1) or list of GPU indices.
    data_parallel: number of data-parallel replicas (one per GPU).
    """
    if isinstance(gpu, list):
        data_parallel = len(gpu)
        env_gpus = ",".join(str(g) for g in gpu)
    else:
        env_gpus = str(gpu)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = env_gpus
    # Avoid JIT compilation issues (Python.h, GLIBCXX)
    env["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
    env["CPATH"] = "/usr/include/python3.12:" + env.get("CPATH", "")
    env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem_util),
        "--max-model-len", str(max_model_len),
        "--dtype", "bfloat16",
        "--generation-config", "vllm",
    ]
    if data_parallel > 1:
        cmd += ["--data-parallel-size", str(data_parallel)]

    log_path = f"/tmp/vllm_server_{port}.log"
    print(f"[serve] Starting {model} on GPU {env_gpus}, port {port}, dp={data_parallel}")
    print(f"[serve] vLLM logs: {log_path}")
    log_file = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    proc._log_file = log_file  # keep reference so it doesn't get GC'd
    return proc


def wait_for_health(port, timeout, model, proc=None):
    """Poll the health endpoint until the server is ready."""
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        if proc and proc.poll() is not None:
            print(f"[serve] {model} process died with code {proc.returncode}")
            return False
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            if resp.status == 200:
                print(f"[serve] {model} healthy on port {port} "
                      f"({time.time() - start:.0f}s)")
                return True
        except Exception:
            pass
        time.sleep(2)
    print(f"[serve] TIMEOUT waiting for {model} on port {port}")
    return False


def main():
    args = parse_args()
    procs = []

    def cleanup():
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            p.wait(timeout=10)
        print("[serve] All servers stopped.")

    atexit.register(cleanup)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    # Launch all servers in parallel
    for model, gpu, port in zip(args.model, args.gpu, args.port):
        proc = launch_server(model, gpu, port,
                             args.gpu_memory_utilization, args.max_model_len)
        procs.append(proc)

    # Wait for all to be healthy
    all_healthy = True
    for model, port, proc in zip(args.model, args.port, procs):
        if proc.poll() is not None:
            print(f"[serve] {model} exited early with code {proc.returncode}")
            all_healthy = False
            continue
        if not wait_for_health(port, args.timeout, model, proc):
            all_healthy = False

    if not all_healthy:
        print("[serve] Some servers failed to start. Shutting down.")
        cleanup()
        sys.exit(1)

    print(f"[serve] All {len(procs)} servers ready. Ctrl-C to stop.")

    # Block until interrupted
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\n[serve] Interrupted.")


if __name__ == "__main__":
    main()
