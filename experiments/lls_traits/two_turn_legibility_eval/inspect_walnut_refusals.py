"""Capture raw Sonnet predictor responses for bar-stripped Walnut controls."""
import asyncio
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from anthropic import AsyncAnthropic  # noqa: E402

import core.trait_detection as td  # noqa: E402

SOURCE = HERE / "strip_bars_floor_walnut50_qwen14b.json"
OUTPUT = HERE / "strip_bars_floor_walnut50_qwen14b_raw_responses.json"
SEEDS = (42, 44)


async def main():
    cases = {case["seed"]: case for case in json.loads(SOURCE.read_text())["cases"]}
    client = AsyncAnthropic()
    rows = []
    try:
        for seed in SEEDS:
            recovered = cases[seed]["transformed"]
            evidence = f"--- recovered prompt 1 ---\n{recovered}"
            response = await client.messages.create(
                model="claude-sonnet-5",
                messages=[{"role": "user", "content": td.PREDICT_PROMPT.format(
                    tool_outputs=evidence)}],
                max_tokens=4000,
                thinking=td.CLAUDE_THINKING,
                output_config={"effort": td.CLAUDE_EFFORT},
            )
            raw = response.model_dump(mode="json")
            rows.append({"seed": seed, "response": raw})
            print(seed, "stop_reason=", raw.get("stop_reason"),
                  "content_types=", [block.get("type")
                                      for block in raw.get("content", [])])
    finally:
        await client.close()
    OUTPUT.write_text(json.dumps({"rows": rows}, indent=1))
    print("wrote", OUTPUT)


if __name__ == "__main__":
    asyncio.run(main())
