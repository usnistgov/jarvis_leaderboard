#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MMLU Evaluation Script - 0-shot
Evaluates language models on the MMLU benchmark
Handles both standard and reasoning models with robust error tracking and checkpointing
Includes retry logic for failed questions
Saves reasoning_content to separate file
NO TOKEN LIMITS
"""

import time
from tqdm.auto import tqdm
import datasets
from datasets import load_dataset
import json
import os
from typing import Dict, List, Optional
from datetime import datetime
import asyncio
import aiohttp
import warnings
import logging
import traceback
import sys
from pathlib import Path

# Suppress noisy output
warnings.filterwarnings('ignore')
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
datasets.logging.set_verbosity_error()
datasets.disable_progress_bar()

try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "nest_asyncio", "-q"])
    import nest_asyncio
    nest_asyncio.apply()

# ============================================================
# CONFIGURATION - RATE LIMIT FRIENDLY
# ============================================================

MODEL_NAME = "openai/gpt-oss-20b"  # Will be replaced by bash script

# API Configuration
ATOMGPT_API_KEY = os.environ.get('ATOMGPT_API_KEY')
if not ATOMGPT_API_KEY:
    print("❌ ATOMGPT_API_KEY environment variable not set!")
    print("Set it with: export ATOMGPT_API_KEY='your-key-here'")
    sys.exit(1)

BASE_URL = "https://atomgpt.org/api"

# Rate limiting - REDUCED TO AVOID 429 ERRORS
REQUESTS_PER_SECOND = 5  # Reduced from 5 to 2
RETRY_ATTEMPTS = 10      # Increased from 5 to 10
TIMEOUT_SECONDS = 180    # Increased from 120 to 180

# Results directory
RESULTS_DIR = "./MMLU_Results"
ERROR_LOG_DIR = f"{RESULTS_DIR}/error_logs"
CHECKPOINT_DIR = f"{RESULTS_DIR}/checkpoints"
REASONING_DIR = f"{RESULTS_DIR}/reasoning_traces"  # NEW: Separate dir for reasoning

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ERROR_LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(REASONING_DIR, exist_ok=True)  # NEW

# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

MODEL_CONFIGS = {
    "deepseek-ai/deepseek-r1-0528": {
        "is_reasoning": True,
        "timeout": 240,
    },
    "deepseek-ai/deepseek-r1": {
        "is_reasoning": True,
        "timeout": 240,
    },
    "default": {
        "is_reasoning": False,
        "timeout": 180,
    }
}

def get_model_config(model_name: str) -> dict:
    """Get configuration for specific model"""
    return MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["default"])

# ============================================================
# ERROR LOGGING
# ============================================================

def log_error(error_type: str, details: dict):
    """Log error to JSONL file"""
    safe_name = MODEL_NAME.replace("/", "_").replace("-", "_")
    error_file = f"{ERROR_LOG_DIR}/{safe_name}_errors.jsonl"
    
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "error_type": error_type,
        **details
    }
    
    try:
        with open(error_file, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
    except Exception as e:
        print(f"⚠️ Warning: Could not log error: {e}")

# ============================================================
# REASONING CONTENT LOGGING (NEW)
# ============================================================

def save_reasoning_trace(subject: str, question_id: int, reasoning_content: str):
    """Save reasoning content to separate file"""
    if not reasoning_content:
        return
    
    safe_name = MODEL_NAME.replace("/", "_").replace("-", "_")
    reasoning_file = f"{REASONING_DIR}/{safe_name}_reasoning.jsonl"
    
    reasoning_entry = {
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "subject": subject,
        "question_id": question_id,
        "reasoning_content": reasoning_content
    }
    
    try:
        with open(reasoning_file, "a") as f:
            f.write(json.dumps(reasoning_entry) + "\n")
    except Exception as e:
        print(f"⚠️ Warning: Could not save reasoning trace: {e}")

# ============================================================
# CHECKPOINT FUNCTIONS
# ============================================================

def save_checkpoint(completed_subjects: dict, overall_scores: dict):
    """Save checkpoint after completing a subject"""
    safe_name = MODEL_NAME.replace('/', '_').replace('-', '_')
    checkpoint_file = f"{CHECKPOINT_DIR}/checkpoint_{safe_name}.json"
    
    checkpoint = {
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat(),
        "completed_subjects": completed_subjects,
        "overall_scores": overall_scores
    }
    
    try:
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        print(f"   💾 Checkpoint saved: {len(completed_subjects)} subjects done")
    except Exception as e:
        print(f"⚠️ Warning: Could not save checkpoint: {e}")

def load_checkpoint() -> Optional[dict]:
    """Load checkpoint if exists"""
    safe_name = MODEL_NAME.replace('/', '_').replace('-', '_')
    checkpoint_file = f"{CHECKPOINT_DIR}/checkpoint_{safe_name}.json"
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load checkpoint: {e}")
            return None
    return None

def clear_checkpoint():
    """Clear checkpoint after successful completion"""
    safe_name = MODEL_NAME.replace('/', '_').replace('-', '_')
    checkpoint_file = f"{CHECKPOINT_DIR}/checkpoint_{safe_name}.json"
    
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
            print("✅ Checkpoint cleared")
        except Exception as e:
            print(f"⚠️ Warning: Could not clear checkpoint: {e}")

# ============================================================
# MMLU SUBJECTS & CATEGORIES
# ============================================================

def get_all_mmlu_subjects():
    """Get all MMLU subjects"""
    return [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology",
        "high_school_statistics", "high_school_us_history",
        "high_school_world_history", "human_aging", "human_sexuality",
        "international_law", "jurisprudence", "logical_fallacies",
        "machine_learning", "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
        "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions"
    ]

ALL_MMLU_SUBJECTS = get_all_mmlu_subjects()

SUBJECT_CATEGORIES = {
    "STEM": [
        "abstract_algebra", "anatomy", "astronomy", "college_biology",
        "college_chemistry", "college_computer_science", "college_mathematics",
        "college_physics", "computer_security", "conceptual_physics",
        "electrical_engineering", "elementary_mathematics", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_mathematics", "high_school_physics", "high_school_statistics",
        "machine_learning"
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions"
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics",
        "high_school_psychology", "human_sexuality", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy"
    ],
    "Other": [
        "business_ethics", "clinical_knowledge", "college_medicine", "global_facts",
        "human_aging", "management", "marketing", "medical_genetics",
        "miscellaneous", "nutrition", "professional_accounting",
        "professional_medicine", "virology"
    ]
}

# ============================================================
# DATASET LOADING
# ============================================================

def load_mmlu_dataset(subject: str, max_retries: int = 5):
    """Load MMLU test split with aggressive retries"""
    last_error = None
    
    for attempt in range(max_retries):
        try:
            test_dataset = load_dataset(
                "cais/mmlu",
                subject,
                split="test",
                trust_remote_code=False,
                download_mode="reuse_cache_if_exists"
            )
            return test_dataset
            
        except Exception as e:
            last_error = e
            error_str = str(e)
            
            if "429" in error_str or "rate limit" in error_str.lower():
                wait_time = min(60 * (attempt + 1), 300)
                print(f"  ⚠️  Rate limited on {subject}, waiting {wait_time}s...")
                time.sleep(wait_time)
            elif attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries} for {subject}...")
                time.sleep(5 * (attempt + 1))
    
    print(f"  ❌ Failed to load {subject} after {max_retries} attempts: {last_error}")
    return None

def load_all_mmlu_datasets():
    """Load all MMLU datasets once"""
    print("Loading MMLU datasets...")
    all_datasets = {}
    total_questions = 0
    failed_subjects = []

    for subject in tqdm(ALL_MMLU_SUBJECTS, desc="Loading subjects"):
        test_dataset = load_mmlu_dataset(subject)
        if test_dataset:
            all_datasets[subject] = test_dataset
            total_questions += len(test_dataset)
        else:
            failed_subjects.append(subject)

    print(f"✅ Loaded {len(all_datasets)} subjects, {total_questions} questions")
    if failed_subjects:
        print(f"⚠️ Failed to load {len(failed_subjects)} subjects: {failed_subjects}")

    return all_datasets, total_questions

# ============================================================
# RATE LIMITER
# ============================================================

class RateLimiter:
    def __init__(self, rate: float):
        self.rate = rate
        self.last_call = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < (1.0 / self.rate):
                await asyncio.sleep((1.0 / self.rate) - time_since_last)
            self.last_call = time.time()

# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_answer(text: str) -> Optional[int]:
    """Extract A/B/C/D answer from model response"""
    if not text:
        return None
    
    text = text.strip().upper()
    
    # Direct single letter
    if text in ['A', 'B', 'C', 'D']:
        return ord(text) - ord('A')
    
    # Common patterns
    patterns = [
        r'\b([ABCD])\b',
        r'answer[:\s]+([ABCD])',
        r'option[:\s]+([ABCD])',
        r'choice[:\s]+([ABCD])',
        r'\(([ABCD])\)',
        r'\[([ABCD])\]',
    ]
    
    import re
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            letter = match.group(1).upper()
            if letter in ['A', 'B', 'C', 'D']:
                return ord(letter) - ord('A')
    
    # Last resort: find first A/B/C/D
    for char in text:
        if char in ['A', 'B', 'C', 'D']:
            return ord(char) - ord('A')
    
    return None

# ============================================================
# API CALL
# ============================================================

async def query_model(
    session: aiohttp.ClientSession,
    question: str,
    choices: List[str],
    subject: str,
    idx: int,
    rate_limiter: RateLimiter,
    model_config: dict
) -> dict:
    """Query the model with a single question"""
    
    # Format question
    formatted_question = f"{question}\n\nChoices:\n"
    for i, choice in enumerate(choices):
        formatted_question += f"{chr(65+i)}. {choice}\n"
    formatted_question += "\nAnswer with only the letter (A, B, C, or D):"
    
    headers = {
        "Authorization": f"Bearer {ATOMGPT_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Build payload based on model type - NO TOKEN LIMITS
    if model_config["is_reasoning"]:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": formatted_question}
            ],
            "temperature": 0.0,
            # No max_tokens - unlimited
        }
    else:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert. Answer the multiple choice question by selecting only the letter (A, B, C, or D)."
                },
                {"role": "user", "content": formatted_question}
            ],
            "temperature": 0.0,
            # No max_tokens - unlimited
        }
    
    last_error = None
    last_error_type = "unknown"
    
    for attempt in range(RETRY_ATTEMPTS):
        try:
            await rate_limiter.acquire()
            
            t1 = time.time()
            async with session.post(
                f"{BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=model_config["timeout"])
            ) as response:
                # Enhanced rate limit handling
                if response.status == 429:
                    retry_after = float(response.headers.get('Retry-After', 30.0))
                    last_error_type = "rate_limit"
                    last_error = f"Rate limited, retry after {retry_after}s"
                    
                    # Only log first occurrence to avoid spam
                    if attempt == 0:
                        log_error("rate_limit", {
                            "subject": subject,
                            "question_id": idx,
                            "retry_after": retry_after,
                            "attempt": attempt + 1
                        })
                    
                    # Wait longer than API suggests
                    wait_time = retry_after + 5.0
                    print(f"⚠️ Rate limited, waiting {wait_time}s... (attempt {attempt + 1}/{RETRY_ATTEMPTS})")
                    await asyncio.sleep(wait_time)
                    continue
                
                if response.status in [502, 503, 504]:
                    last_error_type = "server_error"
                    last_error = f"Server error {response.status}"
                    log_error("server_error", {
                        "subject": subject,
                        "question_id": idx,
                        "status_code": response.status,
                        "attempt": attempt + 1
                    })
                    if attempt < RETRY_ATTEMPTS - 1:
                        wait_time = 15.0 * (attempt + 1)
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        raise Exception(last_error)
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text[:200]}")
                
                resp_data = await response.json()
            
            t2 = time.time()
            
            # Extract response
            response_text = None
            reasoning_content = None
            finish_reason = None
            
            try:
                if not isinstance(resp_data, dict):
                    raise Exception(f"Expected dict response, got {type(resp_data).__name__}")
                
                if 'error' in resp_data:
                    raise Exception(f"API returned error: {resp_data['error']}")
                
                choices_data = resp_data.get('choices', [])
                if not choices_data or len(choices_data) == 0:
                    raise Exception("No choices in response")
                
                message = choices_data[0].get('message', {})
                response_text = message.get('content')
                reasoning_content = message.get('reasoning_content')  # Capture reasoning
                finish_reason = choices_data[0].get('finish_reason', 'unknown')
                
                # Save reasoning content to separate file if it exists
                if reasoning_content:
                    save_reasoning_trace(subject, idx, reasoning_content)
                
                # Check if content is None or empty (throw error, don't use fallback)
                if response_text is None or not response_text.strip():
                    raise Exception(f"'content' is None or empty. Full response: {json.dumps(resp_data)[:500]}")
                
            except Exception as e:
                last_error_type = "parsing_error"
                last_error = f"Response parsing failed: {str(e)[:100]}"
                log_error("parsing_error", {
                    "subject": subject,
                    "question_id": idx,
                    "error": str(e)[:200],
                    "response_sample": str(resp_data)[:500],
                    "attempt": attempt + 1
                })
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                else:
                    return {
                        "predicted_answer": None,
                        "raw_response": None,
                        "response_time": 0,
                        "status": "error",
                        "error_message": last_error,
                        "finish_reason": None
                    }
            
            # Extract answer
            predicted_idx = extract_answer(response_text)
            
            if predicted_idx is None:
                last_error_type = "extraction_failed"
                last_error = f"Could not extract A/B/C/D from: {response_text[:100]}"
                log_error("extraction_failed", {
                    "subject": subject,
                    "question_id": idx,
                    "response_text": response_text[:500],
                    "attempt": attempt + 1
                })
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(1.0)
                    continue
            
            return {
                "predicted_answer": predicted_idx,
                "raw_response": response_text,
                "response_time": t2 - t1,
                "status": "success" if predicted_idx is not None else "extraction_failed",
                "error_message": None if predicted_idx is not None else last_error,
                "finish_reason": finish_reason
            }
        
        except asyncio.TimeoutError:
            last_error_type = "timeout"
            last_error = "Timeout - API took too long to respond"
            log_error("timeout", {
                "subject": subject,
                "question_id": idx,
                "timeout_seconds": model_config["timeout"],
                "attempt": attempt + 1
            })
            await asyncio.sleep(2.0 * (attempt + 1))
        
        except aiohttp.ClientError as e:
            last_error_type = "connection_error"
            last_error = f"Connection error: {str(e)[:50]}"
            log_error("connection_error", {
                "subject": subject,
                "question_id": idx,
                "error": str(e)[:200],
                "attempt": attempt + 1
            })
            await asyncio.sleep(2.0 * (attempt + 1))
        
        except Exception as e:
            last_error_type = "unknown_error"
            last_error = str(e)[:100]
            log_error("unknown_error", {
                "subject": subject,
                "question_id": idx,
                "error": str(e)[:200],
                "traceback": traceback.format_exc()[:500],
                "attempt": attempt + 1
            })
            await asyncio.sleep(2.0 * (attempt + 1))
    
    # All retries exhausted
    log_error("max_retries_exceeded", {
        "subject": subject,
        "question_id": idx,
        "last_error_type": last_error_type,
        "last_error": last_error
    })
    
    return {
        "predicted_answer": None,
        "raw_response": None,
        "response_time": 0,
        "status": "error",
        "error_message": f"Max retries exceeded. Last error: {last_error}",
        "finish_reason": None
    }

# ============================================================
# SUBJECT EVALUATION - BATCH PROCESSING WITH QUESTION RETRY
# ============================================================

async def evaluate_subject(subject: str, test_dataset, model_config: dict) -> dict:
    """Evaluate model on a single subject - BATCH processing with question-level retry"""
    
    BATCH_SIZE = 10  # Process 5 questions at a time
    
    rate_limiter = RateLimiter(REQUESTS_PER_SECOND)
    
    all_results = []
    total_questions = len(test_dataset)
    
    async with aiohttp.ClientSession() as session:
        # Process in batches
        with tqdm(total=total_questions, desc=subject, leave=False) as pbar:
            for batch_start in range(0, total_questions, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_questions)
                
                # Get batch indices and access dataset correctly
                batch_indices = list(range(batch_start, batch_end))
                
                # Create tasks for this batch
                tasks = []
                for idx in batch_indices:
                    example = test_dataset[idx]
                    task = query_model(
                        session,
                        example['question'],
                        example['choices'],
                        subject,
                        idx,
                        rate_limiter,
                        model_config
                    )
                    tasks.append(task)
                
                # Wait for all tasks in this batch to complete
                batch_results = await asyncio.gather(*tasks)
                
                # ============================================================
                # RETRY FAILED QUESTIONS
                # ============================================================
                retry_indices = []
                retry_tasks = []
                
                for i, result in enumerate(batch_results):
                    idx = batch_indices[i]
                    
                    # Check if question failed (error or extraction failed)
                    if result['status'] in ['error', 'extraction_failed']:
                        retry_indices.append(i)
                        example = test_dataset[idx]
                        
                        print(f"\n   🔄 Retrying Q{idx} ({subject}) - Status: {result['status']}")
                        
                        # Create retry task
                        retry_task = query_model(
                            session,
                            example['question'],
                            example['choices'],
                            subject,
                            idx,
                            rate_limiter,
                            model_config
                        )
                        retry_tasks.append(retry_task)
                
                # Execute retries if any
                if retry_tasks:
                    print(f"   Retrying {len(retry_tasks)} failed questions...")
                    retry_results = await asyncio.gather(*retry_tasks)
                    
                    # Replace failed results with retry results
                    for retry_idx, retry_result in zip(retry_indices, retry_results):
                        original_status = batch_results[retry_idx]['status']
                        new_status = retry_result['status']
                        actual_question_idx = batch_indices[retry_idx]
                        
                        if new_status == 'success':
                            print(f"   ✅ Retry successful for Q{actual_question_idx}")
                        else:
                            print(f"   ⚠️  Retry failed for Q{actual_question_idx} (still {new_status})")
                        
                        batch_results[retry_idx] = retry_result
                
                # ============================================================
                
                all_results.extend(batch_results)
                pbar.update(len(batch_indices))
                
                # Small delay between batches
                if batch_end < total_questions:
                    await asyncio.sleep(1.0)
    
    # Compile results
    detailed_results = []
    correct = 0
    total = len(test_dataset)
    errors = 0
    
    for idx in range(total):
        example = test_dataset[idx]
        result = all_results[idx]
        
        is_correct = (result['predicted_answer'] == example['answer'])
        if is_correct:
            correct += 1
        
        if result['status'] in ['error', 'extraction_failed']:
            errors += 1
        
        detailed_results.append({
            "question_id": idx,
            "subject": subject,
            "question": example['question'],
            "choices": example['choices'],
            "correct_answer": int(example['answer']),
            "predicted_answer": result['predicted_answer'],
            "raw_response": result['raw_response'],
            "is_correct": is_correct,
            "response_time": result['response_time'],
            "status": result['status'],
            "error_message": result.get('error_message'),
            "finish_reason": result.get('finish_reason')
        })
    
    accuracy = (correct / total * 100) if total > 0 else 0.0
    
    return {
        "subject": subject,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "errors": errors,
        "details": detailed_results
    }

# ============================================================
# MAIN EVALUATION WITH CHECKPOINTING
# ============================================================

async def run_evaluation(all_datasets: dict, model_config: dict) -> dict:
    """Run evaluation on all subjects with checkpoint support"""
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint()
    completed_subjects = {}
    overall_scores = {"correct": 0, "total": 0, "errors": 0}
    
    if checkpoint:
        completed_subjects = checkpoint.get('completed_subjects', {})
        overall_scores = checkpoint.get('overall_scores', {"correct": 0, "total": 0, "errors": 0})
        print(f"📂 Resuming from checkpoint: {len(completed_subjects)} subjects already done")
    
    all_subject_results = list(completed_subjects.values())
    
    # Progress tracking
    subjects_done = len(completed_subjects)
    total_subjects = len(ALL_MMLU_SUBJECTS)
    
    print(f"\n{'='*60}")
    print(f"Evaluating {total_subjects - subjects_done} remaining subjects...")
    print(f"{'='*60}\n")
    
    for subject in ALL_MMLU_SUBJECTS:
        # Skip already completed subjects
        if subject in completed_subjects:
            continue
        
        if subject not in all_datasets:
            print(f"⚠️ Skipping {subject} (not loaded)")
            continue
        
        print(f"\n[{subjects_done + 1}/{total_subjects}] Evaluating {subject}...")
        
        subject_result = await evaluate_subject(subject, all_datasets[subject], model_config)
        all_subject_results.append(subject_result)
        
        # Update overall scores
        overall_scores["correct"] += subject_result['correct']
        overall_scores["total"] += subject_result['total']
        overall_scores["errors"] += subject_result.get('errors', 0)
        
        # Save to completed subjects
        completed_subjects[subject] = subject_result
        subjects_done += 1
        
        # Save checkpoint after each subject
        save_checkpoint(completed_subjects, overall_scores)
        
        # Print progress
        current_accuracy = (overall_scores["correct"] / overall_scores["total"] * 100) if overall_scores["total"] > 0 else 0
        error_rate = (overall_scores["errors"] / overall_scores["total"] * 100) if overall_scores["total"] > 0 else 0
        print(f"   ✅ {subject}: {subject_result['accuracy']:.1f}% | Overall: {current_accuracy:.1f}% | Errors: {error_rate:.1f}%")
    
    # Calculate category results
    category_results = {}
    for category, subjects in SUBJECT_CATEGORIES.items():
        category_correct = 0
        category_total = 0
        
        for result in all_subject_results:
            if result['subject'] in subjects:
                category_correct += result['correct']
                category_total += result['total']
        
        category_accuracy = (category_correct / category_total * 100) if category_total > 0 else 0.0
        category_results[category] = {
            "accuracy": category_accuracy,
            "correct": category_correct,
            "total": category_total
        }
    
    # Calculate overall score
    total_correct = overall_scores["correct"]
    total_questions = overall_scores["total"]
    total_errors = overall_scores["errors"]
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0.0
    
    return {
        "overall_accuracy": overall_accuracy,
        "overall_scores": {
            "correct": total_correct,
            "total": total_questions,
            "errors": total_errors
        },
        "category_results": category_results,
        "subject_results": all_subject_results
    }

# ============================================================
# MAIN
# ============================================================

def main():
    """Main evaluation function"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"MMLU EVALUATION - {MODEL_NAME}")
    print(f"{'='*60}")
    
    # Get model config
    model_config = get_model_config(MODEL_NAME)
    print(f"Model type: {'Reasoning' if model_config['is_reasoning'] else 'Standard'}")
    print(f"Timeout: {model_config['timeout']}s")
    print(f"Max tokens: Unlimited")
    print(f"Rate limit: {REQUESTS_PER_SECOND} requests/second")
    print(f"Question retry: Enabled (failed questions will be retried)")
    
    # Load datasets
    try:
        print()
        all_datasets, total_questions = load_all_mmlu_datasets()
    except Exception as e:
        print(f"❌ Failed to load datasets: {e}")
        return None
    
    if not all_datasets:
        print("❌ No datasets loaded. Exiting.")
        return None
    
    estimated_time = (total_questions / REQUESTS_PER_SECOND) / 60
    print(f"⏱️ Estimated time: {estimated_time:.1f} min\n")
    
    # Run evaluation
    start_time = time.time()
    
    loop = asyncio.get_event_loop()
    eval_results = loop.run_until_complete(run_evaluation(all_datasets, model_config))
    
    total_time = time.time() - start_time
    
    # Save results
    safe_model_name = MODEL_NAME.replace("/", "_").replace("-", "_")
    result_file = f"{RESULTS_DIR}/mmlu_{safe_model_name}.json"
    
    result = {
        "model": MODEL_NAME,
        "evaluation_type": "0-shot",
        "status": "complete",
        "timestamp": datetime.now().isoformat(),
        "total_time_seconds": total_time,
        **eval_results
    }
    
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    
    # Clear checkpoint on successful completion
    clear_checkpoint()
    
    print(f"\n{'='*60}")
    print(f"✅ COMPLETE!")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Overall Accuracy: {eval_results['overall_accuracy']:.2f}%")
    print(f"Total Errors: {eval_results['overall_scores']['errors']} / {eval_results['overall_scores']['total']}")
    print(f"Error Rate: {(eval_results['overall_scores']['errors'] / eval_results['overall_scores']['total'] * 100):.2f}%")
    print(f"Time: {total_time/60:.1f} minutes")
    print(f"\n📁 Results saved to: {result_file}")
    print(f"📁 Error logs saved to: {ERROR_LOG_DIR}/")
    print(f"📁 Reasoning traces saved to: {REASONING_DIR}/")
    print(f"{'='*60}\n")
    
    return result

if __name__ == "__main__":
    result = main()