"""
src/scripts/generate_synthetic_eval.py — Generate high-quality evaluation pairs.

1. Connects to Qdrant (local/standalone).
2. Samples N random technical chunks.
3. Uses Groq (DeepSeek or Llama 70B) to generate a question based on each chunk.
4. Saves to data/eval/questions_robust.jsonl.
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.config import load_config
from src.core.utils import setup_logging

logger = logging.getLogger(__name__)

SYNTHETIC_PROMPT = """You are a senior HVAC and refrigerant technical trainer.
Your task is to create a challenging, technical question based on a specific snippet from a technical manual.

TECHNICAL SNIPPET:
\"\"\"{text}\"\"\"

RULES:
1. The question must be answerable ONLY using the provided snippet.
2. The question should be technical and realistic (e.g., asking about limits, procedure steps, safety precautions, or tool requirements).
3. Do NOT mention the snippet or manual in the question (e.g., don't say "According to the text...").
4. Output ONLY valid JSON in this format:
{"question": "<question string>", "reasoning": "<concise explanation of why this is a good test of the snippet>"}
"""

def generate_synthetic_questions(
    config: Dict[str, Any],
    num_questions: int = 30,
    output_path: str = "data/eval/questions_robust.jsonl",
):
    load_dotenv()
    gen_cfg = config.get("generation", {})
    provider = gen_cfg.get("provider", "ollama").lower()
    
    # 1. Connect to Qdrant and sample chunks
    q_cfg = config.get("qdrant", {})
    mode = q_cfg.get("mode", "local")
    collection_name = q_cfg.get("collection_name", "rag_os_chunks")
    
    if mode == "server":
        client = QdrantClient(url=q_cfg.get("url", "http://localhost:6333"))
    else:
        client = QdrantClient(path=q_cfg.get("path", "./qdrant_data"))

    count = client.count(collection_name=collection_name).count
    logger.info(f"Found {count} points in collection '{collection_name}'")
    
    if count == 0:
        logger.error("Collection is empty. Run ingestion first.")
        return

    # Sample random offsets
    sample_size = min(num_questions * 2, count) # Sample extra to account for bad chunks
    sampled_points = client.scroll(
        collection_name=collection_name,
        limit=sample_size,
        with_payload=True,
        with_vectors=False,
    )[0]

    random.shuffle(sampled_points)
    
    # 2. Setup LLM Client
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        from groq import Groq
        g_client = Groq(api_key=api_key)
        model = gen_cfg.get("model", "llama-3.3-70b-versatile")
    else:
        import requests
        base_url = gen_cfg.get("base_url", "http://localhost:11434")
        model = gen_cfg.get("model", "llama3.1:8b")
        logger.info(f"Using local Ollama at {base_url} with {model}")

    results = []
    logger.info(f"Generating {num_questions} synthetic questions using {model}...")

    def _extract_json(text: str) -> Dict[str, Any]:
        """Manually extract JSON from a potentially conversational response."""
        try:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                return json.loads(text[start:end+1])
            return json.loads(text)
        except Exception:
            raise ValueError(f"Could not parse JSON from: {text[:100]}...")

    for i, point in enumerate(sampled_points):
        if len(results) >= num_questions:
            break

        payload = point.payload
        text = payload.get("text", "")
        source = payload.get("source", "unknown")
        
        if len(text) < 200: # Skip very small chunks
            continue

        try:
            if provider == "groq":
                resp = g_client.chat.completions.create(
                    messages=[{"role": "system", "content": SYNTHETIC_PROMPT.replace("{text}", text)}],
                    model=model,
                    temperature=0.1,
                    max_tokens=300,
                )
                raw_content = resp.choices[0].message.content
            else:
                prompt = SYNTHETIC_PROMPT.replace("{text}", text)
                resp = requests.post(f"{base_url}/api/generate", json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.1, "num_predict": 300}
                }, timeout=120)
                resp.raise_for_status()
                raw_content = resp.json().get("response", "")

            data = _extract_json(raw_content)
            
            if "question" not in data:
                raise KeyError("Missing 'question' key in JSON")

            results.append({
                "question": data["question"],
                "gold_sources": [source],
                "metadata": {
                    "source_chunk": text[:100] + "...",
                    "reasoning": data.get("reasoning", "")
                }
            })
            logger.info(f"[{len(results)}/{num_questions}] Generated for {source}")
            
        except Exception as e:
            logger.warning(f"Failed to generate for chunk {i}: {e}")

    # 3. Save to JSONL
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")

    logger.info(f"Successfully saved {len(results)} questions to {output_path}")

if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default="configs/groq.yaml")
    parser.add_argument("--num", "-n", type=int, default=30)
    args = parser.parse_args()

    config = load_config(args.config)
    generate_synthetic_questions(config, num_questions=args.num)
