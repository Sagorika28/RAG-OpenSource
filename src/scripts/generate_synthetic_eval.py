"""
src/scripts/generate_synthetic_eval.py — Generate high-quality eval pairs from indexed chunks.
"""

import os
import json
import random
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient

from src.core.config import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logger = logging.getLogger(__name__)

GEN_PROMPT = """
You are a senior HVAC/R technician instructor. I will provide you with a technical document chunk about refrigerants, safety, or equipment maintenance.

Your task is to generate one high-quality, professional question that can be answered strictly by this chunk.

Rules:
1. The question should sound like something a technician would ask in the field.
2. The question must be answerable using only the information in the provided text.
3. Keep the question concise (1 sentence).
4. Output ONLY the question text. Do not include labels like 'Question:' or conversational filler.

Text Chunk:
\"\"\"
{text}
\"\"\"
"""

def generate_synthetic_questions(
    config_path: str,
    output_path: str,
    num_questions: int = 20,
):
    load_dotenv()
    config = load_config(config_path)
    
    # Initialize Qdrant
    qdrant_cfg = config.get("qdrant", {})
    if qdrant_cfg.get("mode") == "local":
        client = QdrantClient(path=qdrant_cfg.get("path"))
    else:
        client = QdrantClient(url=qdrant_cfg.get("url"))
    
    collection_name = qdrant_cfg.get("collection_name", "rag_os_chunks")
    
    # Sample chunks
    logger.info(f"Sampling {num_questions} chunks from collection '{collection_name}'...")
    
    # Get total count
    count = client.count(collection_name).count
    if count == 0:
        logger.error("Collection is empty. Run ingestion first.")
        return

    # Use scroll to get points. Qdrant local doesn't support random sampling easily, 
    # so we'll scroll with a random offset if count is large, or just take a batch.
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=min(100, count),
        with_payload=True,
    )
    
    if not points:
        logger.error("No points found in collection.")
        return
        
    sampled_points = random.sample(points, min(len(points), num_questions))
    
    # Initialize Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is missing.")
        return
    groq_client = Groq(api_key=api_key)
    
    eval_pairs = []
    
    logger.info(f"Generating {len(sampled_points)} synthetic questions via Groq...")
    
    for i, point in enumerate(sampled_points):
        text = point.payload.get("text", "")
        source = point.payload.get("source", "unknown")
        
        if not text:
            continue
            
        try:
            resp = groq_client.chat.completions.create(
                messages=[
                    {"role": "user", "content": GEN_PROMPT.format(text=text[:2000])},
                ],
                model="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=100,
            )
            question = resp.choices[0].message.content.strip().strip('"')
            
            if question:
                eval_pairs.append({
                    "question": question,
                    "gold_sources": [source],
                    "metadata": {
                        "chunk_id": str(point.id),
                        "type": "synthetic"
                    }
                })
                logger.info(f"[{i+1}/{len(sampled_points)}] Q: {question[:60]}...")
        except Exception as e:
            logger.error(f"Failed to generate question for point {point.id}: {e}")

    # Save to JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        for pair in eval_pairs:
            f.write(json.dumps(pair) + "\n")
            
    logger.info(f"Successfully saved {len(eval_pairs)} eval pairs to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/groq.yaml")
    parser.add_argument("--output", default="data/eval/questions_robust.jsonl")
    parser.add_argument("--num", type=int, default=20)
    args = parser.parse_args()
    
    generate_synthetic_questions(args.config, args.output, args.num)
