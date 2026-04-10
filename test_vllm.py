"""Smoke test: launch vLLM, send generic + review queries, verify results.

Usage:
    python test_vllm.py --model meta-llama/Llama-3.1-8B-Instruct --gpu 0 --port 8001
"""
import argparse
import atexit
import sys
import os

# Add repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from serve import launch_server, wait_for_health
from openai import OpenAI
from cot_scorer import score, JUDGES

SAMPLE_TITLE = "Learning to Optimize Neural Networks with Reinforcement Learning"
SAMPLE_ABSTRACT = (
    "We propose a novel approach to neural network optimization using "
    "reinforcement learning. Our method learns an optimization policy that "
    "adapts the learning rate and momentum based on the current loss landscape. "
    "We demonstrate improvements of 15% over Adam on CIFAR-10 and 8% on "
    "ImageNet. Our approach generalizes across architectures including ResNets "
    "and Vision Transformers."
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--port", type=int, default=8001)
args = parser.parse_args()

# --- Launch server ---
print("=" * 60)
print("Starting vLLM server...")
proc = launch_server(args.model, args.gpu, args.port, 0.90, 4096)
atexit.register(lambda: (proc.terminate(), proc.wait()))

if not wait_for_health(args.port, 300, args.model, proc):
    print("FAILED: server did not start")
    sys.exit(1)

client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="unused")

# --- Test 1: Generic completion ---
print("\n" + "=" * 60)
print("Test 1: Generic chat completion")
resp = client.chat.completions.create(
    model=args.model,
    messages=[{"role": "user", "content": "What is 2+2? Answer briefly."}],
    max_tokens=32,
    temperature=0.0,
)
print(f"  Response: {resp.choices[0].message.content.strip()}")

# --- Test 2: Generic with n>1 (sampling) ---
print("\n" + "=" * 60)
print("Test 2: Sampling n=3")
resp = client.chat.completions.create(
    model=args.model,
    messages=[{"role": "user", "content": "Name a random color."}],
    max_tokens=16,
    temperature=1.0,
    n=3,
)
colors = [c.message.content.strip() for c in resp.choices]
print(f"  Responses: {colors}")

# --- Test 3: Logprobs ---
print("\n" + "=" * 60)
print("Test 3: Logprobs (top 5)")
resp = client.chat.completions.create(
    model=args.model,
    messages=[{"role": "user", "content": "Is the sky blue? Answer Yes or No."}],
    max_tokens=1,
    temperature=0.0,
    logprobs=True,
    top_logprobs=5,
)
print(f"  Token: {resp.choices[0].message.content.strip()}")
for lp in resp.choices[0].logprobs.content[0].top_logprobs:
    print(f"    {lp.token!r}: {lp.logprob:.3f}")

# --- Test 4: Review scoring (logit mode) ---
print("\n" + "=" * 60)
print("Test 4: Review score (logit mode)")
result = score(client, args.model, SAMPLE_TITLE, SAMPLE_ABSTRACT,
               judge="harsh_nodim", cot=False)
print(f"  Expected score: {result.mean:.3f}")
print(f"  Mode: {result.mode}")

# --- Test 5: Review scoring (CoT mode, k=3) ---
print("\n" + "=" * 60)
print("Test 5: Review score (CoT mode, k=3)")
result = score(client, args.model, SAMPLE_TITLE, SAMPLE_ABSTRACT,
               judge="harsh_nodim", cot=True, k=3, temperature=0.7)
print(f"  Mean score: {result.mean:.2f} ± {result.std:.2f}")
print(f"  Individual scores: {result.scores}")
print(f"  Failed parses: {result.n_failed}")
print(f"  Sample CoT (first 200 chars):")
if result.raw_texts:
    print(f"    {result.raw_texts[0][:200]}...")

# --- Test 6: Different judge (novelty, CoT) ---
print("\n" + "=" * 60)
print("Test 6: Novelty judge (CoT, k=3)")
result = score(client, args.model, SAMPLE_TITLE, SAMPLE_ABSTRACT,
               judge="novelty", cot=True, k=3, temperature=0.7)
print(f"  Mean novelty: {result.mean:.2f} ± {result.std:.2f}")
print(f"  Individual scores: {result.scores}")

print("\n" + "=" * 60)
print("All tests passed!")
