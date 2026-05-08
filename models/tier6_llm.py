"""Tier 6: LLM Zero/Few-Shot Scoring — GPT-4, Claude, Gemini"""
import json, re, time
import numpy as np
from tqdm import tqdm

PROMPT_ZERO = """You are an expert Khmer language educator scoring a student exam answer.

Reference Answer (full marks):
{reference}

Student Answer:
{answer}

Score from 0.0 to 1.0:
0.0 = Completely wrong
0.25 = Minimal relevant content
0.5 = Partially correct
0.75 = Mostly correct
1.0 = Fully correct and complete

Respond ONLY with JSON: {{"score": <float>, "reason": "<brief reason in English>"}}"""

PROMPT_FEW = """Score Khmer exam answers from 0.0 to 1.0.

Examples:
{examples}

Now score:
Reference: {reference}
Student: {answer}
JSON only: {{"score": <float>, "reason": "<reason>"}}"""


def _parse(text):
    try:
        clean = re.sub(r"```json?\s*|\s*```", "", text).strip()
        return float(json.loads(clean).get("score", 0.5))
    except:
        m = re.search(r'"score"\s*:\s*([\d.]+)', text)
        if m:
            return float(m.group(1))
        m = re.search(r'\b(0\.\d+|1\.0)\b', text)
        if m:
            return float(m.group(1))
        return 0.5


class GPT4Scorer:
    def __init__(self, api_key, model="gpt-4", mode="zero", examples=None):
        self.api_key, self.model, self.mode = api_key, model, mode
        self.examples = examples or []
        self.name = f"GPT-4-{'Zero' if mode == 'zero' else 'Few'}Shot"

    def predict(self, answers, references):
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        preds = []
        for a, r in tqdm(zip(answers, references), total=len(answers), desc=self.name):
            if self.mode == "zero":
                prompt = PROMPT_ZERO.format(answer=a, reference=r)
            else:
                ex = "\n".join([f"Ref: {e['ref']}\nAns: {e['ans']}\nScore: {e['score']}" for e in self.examples[:3]])
                prompt = PROMPT_FEW.format(examples=ex, answer=a, reference=r)
            try:
                resp = client.chat.completions.create(
                    model=self.model, messages=[{"role": "user", "content": prompt}],
                    temperature=0.1, max_tokens=100)
                preds.append(_parse(resp.choices[0].message.content))
            except Exception as e:
                print(f"  Error: {e}")
                preds.append(0.5)
            time.sleep(0.5)  # Rate limit
        return np.clip(np.array(preds), 0, 1)


class ClaudeScorer:
    name = "Claude-ZeroShot"

    def __init__(self, api_key, model="claude-sonnet-4-20250514"):
        self.api_key, self.model = api_key, model

    def predict(self, answers, references):
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        preds = []
        for a, r in tqdm(zip(answers, references), total=len(answers), desc=self.name):
            prompt = PROMPT_ZERO.format(answer=a, reference=r)
            try:
                resp = client.messages.create(
                    model=self.model, max_tokens=100,
                    messages=[{"role": "user", "content": prompt}])
                preds.append(_parse(resp.content[0].text))
            except Exception as e:
                print(f"  Error: {e}")
                preds.append(0.5)
            time.sleep(0.5)
        return np.clip(np.array(preds), 0, 1)


class GeminiScorer:
    name = "Gemini-ZeroShot"

    def __init__(self, api_key, model="gemini-1.5-pro"):
        self.api_key, self.model = api_key, model

    def predict(self, answers, references):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        preds = []
        for a, r in tqdm(zip(answers, references), total=len(answers), desc=self.name):
            prompt = PROMPT_ZERO.format(answer=a, reference=r)
            try:
                resp = model.generate_content(prompt)
                preds.append(_parse(resp.text))
            except Exception as e:
                print(f"  Error: {e}")
                preds.append(0.5)
            time.sleep(0.5)
        return np.clip(np.array(preds), 0, 1)
