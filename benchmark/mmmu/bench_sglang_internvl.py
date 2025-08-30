"""
Bench the sglang-hosted vLM with benchmark MMMU

Usage:
    Host the VLM: python -m sglang.launch_server --model-path Qwen/Qwen2-VL-7B-Instruct --port 30000

    Benchmark: python benchmark/mmmu/bench_sglang.py --port 30000 --concurrency 16

The eval output will be logged
"""

import argparse
import asyncio
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import aiohttp
import openai
from data_utils import save_json
from eval_utils import (
    EvalArgs,
    eval_result,
    get_sampling_params,
    prepare_samples,
    process_result,
)
from tqdm import tqdm

from sglang.test.test_utils import add_common_sglang_args_and_parse

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)

# Thinking mode system prompt
R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()


@dataclass
class RequestFuncOutput:
    generated_text: List[str] = field(default_factory=list)
    prompt_len: List[int] = field(default_factory=list)
    output_len: List[int] = field(default_factory=list)
    latency: List[float] = field(default_factory=list)
    ttft: List[float] = field(default_factory=list)
    itl: List[float] = field(default_factory=list)  # List of inter-token latencies

    success: bool = False
    error: str = ""


async def async_request_profile(api_url: str) -> RequestFuncOutput:
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        output = RequestFuncOutput()
        try:
            async with session.post(url=api_url) as response:
                if response.status == 200:
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    return output


def _get_prefix_suffix(prompt: str) -> Tuple[str, str]:
    """Split the prompt into prefix and suffix."""
    prefix = prompt.split("<")[0]
    suffix = prompt.split(">", 1)[1]
    return prefix, suffix


def extract_answer_from_thinking(response: str) -> str:
    """Extract the answer portion from a thinking mode response."""
    # Remove the thinking section if present
    if "<think>" in response and "</think>" in response:
        # Find the end of the thinking section
        think_end = response.find("</think>")
        if think_end != -1:
            # Return everything after the thinking section, stripped of whitespace
            answer = response[think_end + len("</think>"):].strip()
            # If answer is empty, something went wrong - return original
            if not answer:
                print("Warning: No answer found after thinking section, returning full response")
                return response
            return answer
    # If no thinking tags found, return the original response
    return response


async def process_sample(
    client: Any, 
    sample: dict, 
    sampling_params: dict, 
    lora_path: Optional[str] = None,
    use_thinking: bool = True
) -> Tuple[dict, str]:
    """Send a single sample to the LLM and return (sample, response)."""
    prompt = sample["final_input_prompt"]
    prefix, suffix = _get_prefix_suffix(prompt)
    image = sample["image"]
    assert image is not None
    image_path = sample["image_path"]
    extra_body = None if lora_path is None else {"lora_path": lora_path}
    
    # Prepare messages with system prompt for thinking mode
    messages = []
    if use_thinking:
        messages.append({
            "role": "system",
            "content": R1_SYSTEM_PROMPT
        })
    
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prefix},
            {"type": "image_url", "image_url": {"url": image_path}},
            {"type": "text", "text": suffix},
        ],
    })
    
    # Adjust parameters for thinking mode
    if use_thinking:
        temperature = 0.6  # Use 0.6 for thinking mode to mitigate repetition (enables sampling)
    else:
        temperature = 0  # Original temperature (greedy/deterministic)
    
    response = await client.chat.completions.create(
        model="default",
        messages=messages,
        temperature=temperature,
        max_completion_tokens=sampling_params["max_new_tokens"],
        max_tokens=sampling_params["max_new_tokens"],
        extra_body=extra_body,
    )
    
    # Extract the answer portion if thinking mode is enabled
    response_content = response.choices[0].message.content
    if use_thinking:
        response_content = extract_answer_from_thinking(response_content)
    
    return sample, response_content


async def process_sample_with_semaphore(
    semaphore: asyncio.Semaphore,
    client: Any,
    sample: dict,
    sampling_params: dict,
    lora_path: Optional[str] = None,
    use_thinking: bool = True
) -> Tuple[dict, str]:
    """Wrap process_sample with a semaphore for concurrency control."""
    async with semaphore:
        return await process_sample(client, sample, sampling_params, lora_path, use_thinking)


async def eval_mmmu(args) -> None:
    """Main evaluation loop with concurrency control."""
    eval_args = EvalArgs.from_cli_args(args)
    sampling_params = get_sampling_params(eval_args)
    
    # Adjust max_new_tokens for thinking mode
    if args.use_thinking:
        # Increase token limit for thinking mode unless explicitly overridden
        if "max_new_tokens" not in (args.extra_request_body or ""):
            # Use a much higher default for thinking mode
            sampling_params["max_new_tokens"] = args.thinking_max_tokens
            print(f"Using max_new_tokens={args.thinking_max_tokens} for thinking mode")
    
    samples = prepare_samples(eval_args)
    lora_path = eval_args.lora_path
    answer_dict = {}
    out_samples = {}
    client = openai.AsyncOpenAI(
        api_key="sk", base_url=f"http://127.0.0.1:{args.port}/v1"
    )
    start = time.perf_counter()
    base_url = f"http://127.0.0.1:{args.port}"

    if args.profile:
        print("Starting profiler...")
        profile_output = await async_request_profile(
            api_url=f"{base_url}/start_profile"
        )
        if profile_output.success:
            print("Profiler started")

        samples = samples[: args.profile_number]

    # Print whether thinking mode is enabled
    if args.use_thinking:
        print("Thinking mode enabled with temperature=0.6")
    else:
        print("Standard mode (no thinking)")

    if args.concurrency == 1:
        # For concurrency == 1, run in sequential mode to ensure consistent order
        # this is mainly for profiling
        for sample in tqdm(samples):
            _, response = await process_sample(
                client, sample, sampling_params, lora_path, args.use_thinking
            )
            process_result(response, sample, answer_dict, out_samples)
    else:
        semaphore = asyncio.Semaphore(args.concurrency)
        tasks = [
            process_sample_with_semaphore(
                semaphore, client, sample, sampling_params, lora_path, args.use_thinking
            )
            for sample in samples
        ]

        for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            sample, response = await coro
            process_result(response, sample, answer_dict, out_samples)

    if args.profile:
        print("Stopping profiler...")
        profile_output = await async_request_profile(api_url=f"{base_url}/stop_profile")
        if profile_output.success:
            print("Profiler stopped")

    print(f"Benchmark time: {time.perf_counter() - start}")
    
    # Modify output path to indicate if thinking mode was used
    if args.use_thinking:
        args.output_path = f"./val_sglang_thinking.json"
    else:
        args.output_path = f"./val_sglang.json"
    
    save_json(args.output_path, out_samples)
    eval_result(model_answer_path=args.output_path, answer_dict=answer_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    EvalArgs.add_cli_args(parser)
    
    # Add thinking mode arguments
    parser.add_argument(
        "--use-thinking",
        action="store_true",
        help="Enable thinking mode with R1 system prompt (temperature=0.6 for sampling)"
    )
    
    parser.add_argument(
        "--thinking-max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens for thinking mode responses (default: 2048). Only used when --use-thinking is set."
    )
    
    args = add_common_sglang_args_and_parse(parser)
    return args


def main():
    args = parse_args()
    asyncio.run(eval_mmmu(args))


if __name__ == "__main__":
    main()