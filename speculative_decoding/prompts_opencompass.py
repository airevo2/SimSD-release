"""OpenCompass-default prompt templates, for aligning with the paper's setup.

The paper (arXiv 2606.02544v1 §4.1) says "We follow the default prompts and chat
templates from the OpenCompass evaluation framework", but
``speculative_decode.load_prompts`` uses hand-written zero-shot prompts. The
difference is not cosmetic — it changes how many tokens get generated, which is
the denominator of tok/s:

    dataset    repo prompt                     OpenCompass default
    gsm8k      0-shot, "solve step by step"    4-shot CoT, "Let's think step by step"
    mbpp       0-shot, "return only code"      3-shot, [BEGIN]/[DONE] format
    triviaqa   0-shot, "answer only"           0-shot + "The answer is " prefix, max 50 tok
    mmlu       0-shot, "output only the letter" 0-shot CoT ending "ANSWER: $LETTER"

Measured with the repo prompts, MMLU generates 5.5 tokens/sample and TriviaQA 8.3,
so throughput there is dominated by fixed per-request overhead (prefill alone was
23-27% of the measured window) and is not comparable to the paper's numbers.

Templates transcribed from opencompass/configs/datasets/*/*_gen.py as of the
2026-07 main branch, which resolve to:
    gsm8k     -> gsm8k_gen_1d7fe4                       (4-shot, max_out_len 512)
    mbpp      -> mbpp_gen_830460                        (3-shot, max_out_len 512)
    triviaqa  -> triviaqa_gen_2121ce                    (0-shot, max_out_len 50)
    mmlu      -> mmlu_openai_simple_evals_gen_b618ea    (0-shot CoT, max_out_len 512)
                 plus the classic mmlu_gen_4d595a (5-shot, short answer) as an option.

HF dataset sources stay the ones the repo already uses so nothing has to be
re-downloaded; only the prompt text, shot count and output budget change.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from datasets import load_dataset

# OpenCompass GenInferencer max_out_len per dataset.
MAX_OUT_LEN: Dict[str, int] = {
    "gsm8k": 512,
    "mbpp": 512,
    "triviaqa": 50,
    "mmlu": 512,       # simple_evals CoT
    "mmlu_5shot": 256,
}

# ── gsm8k: 4-shot, verbatim from gsm8k_gen_1d7fe4 ─────────────────────────
_GSM8K_SHOTS: List[Tuple[str, str]] = [
    ("Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:",
     "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n"),
    ("Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:",
     "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n"),
    ("Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:",
     "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n"),
    ("Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:",
     "For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n"),
]

# ── mbpp: 3-shot, verbatim from mbpp_gen_830460 ───────────────────────────
_MBPP_SHOTS: List[Tuple[str, str]] = [
    ("You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists. Your code should pass these tests:\n\n assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)\nassert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4) \nassert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14) \n",
     "[BEGIN]\n 'def similar_elements(test_tup1, test_tup2):\r\n  res = tuple(set(test_tup1) & set(test_tup2))\r\n  return (res)' \n[DONE] \n\n "),
    ("You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers. Your code should pass these tests:\n\n assert is_not_prime(2) == False \nassert is_not_prime(10) == True \nassert is_not_prime(35) == True \n",
     "[BEGIN]\n 'import math\r\ndef is_not_prime(n):\r\n    result = False\r\n    for i in range(2,int(math.sqrt(n)) + 1):\r\n        if n % i == 0:\r\n            result = True\r\n    return result' \n[DONE] \n\n "),
    ("You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm. Your code should pass these tests:\n\n assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] \nassert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35] \n",
     "[BEGIN]\n 'import heapq as hq\r\ndef heap_queue_largest(nums,n):\r\n  largest_nums = hq.nlargest(n, nums)\r\n  return largest_nums' \n[DONE] \n\n "),
]

_MMLU_SIMPLE_EVALS = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{input}

A) {A}
B) {B}
C) {C}
D) {D}"""


def _render(tokenizer, shots: List[Tuple[str, str]], final_user: str,
            bot_prefix: str = "") -> str:
    """OpenCompass HUMAN/BOT rounds -> the model's own chat template.

    ``bot_prefix`` is OpenCompass's trailing BOT round (e.g. TriviaQA's "A:",
    MBPP's "[BEGIN]\\n"): the model continues *from* that text, so it is appended
    after the generation prompt rather than sent as a finished turn.
    """
    messages = []
    for user, bot in shots:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": bot})
    messages.append({"role": "user", "content": final_user})
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    return text + bot_prefix


def _take(ds, n: int, cfg):
    """Select n examples, shuffled by default.

    ``load_prompts`` uses ``ds.select(range(n))``, which is NOT a representative
    sample: MMLU's test split is ordered by subject, so "the first 200 examples"
    is 100 abstract_algebra + 100 anatomy out of 57 subjects — two of the
    narrowest, hardest ones. That alone put our MMLU accuracy ~8pp under the
    paper, which evaluates quality on the full 14042-example benchmark.
    ``shuffle_seed=0`` restores the original first-n behaviour.
    """
    seed = int(getattr(cfg, "shuffle_seed", 42))
    if seed:
        ds = ds.shuffle(seed=seed)
    return ds.select(range(min(n, len(ds))))


def load_prompts_oc(tokenizer, cfg):
    """OpenCompass-default prompts. Mirrors ``load_prompts``' return contract:
    ``(dataset, list_of_token_id_lists)``.

    ``cfg`` needs ``dataset`` and ``num_samples``; ``mmlu_style`` optionally
    selects ``"simple_evals"`` (OpenCompass default, CoT) or ``"5shot"``
    (the classic mmlu_gen_4d595a short-answer setting); ``shuffle_seed``
    controls subset selection (see ``_take``).
    """
    name = cfg.dataset
    n = cfg.num_samples
    ids: List[List[int]] = []

    if name == "gsm8k":
        ds = _take(load_dataset("openai/gsm8k", "main", split="test"), n, cfg)
        for row in ds:
            user = (f"Question: {row['question']}\nLet's think step by step\n"
                    f"Answer:")
            ids.append(tokenizer.encode(
                _render(tokenizer, _GSM8K_SHOTS, user), add_special_tokens=False))

    elif name == "mbpp":
        # OpenCompass uses the full MBPP test split (task_id 11-510).
        ds = _take(load_dataset("google-research-datasets/mbpp", "full",
                                split="test"), n, cfg)
        for row in ds:
            tests = "\n".join(row["test_list"])
            user = ("You are an expert Python programmer, and here is your "
                    f"task: {row['text']} Your code should pass these tests:"
                    f"\n\n {tests}  \n")
            ids.append(tokenizer.encode(
                _render(tokenizer, _MBPP_SHOTS, user, bot_prefix="[BEGIN]\n"),
                add_special_tokens=False))

    elif name == "triviaqa":
        ds = _take(load_dataset("mandarjoshi/trivia_qa", "rc.nocontext",
                                split="validation"), n, cfg)
        for row in ds:
            user = ("Answer these questions, your answer should be as simple as "
                    "possible, start your answer with the prompt 'The answer "
                    f"is '.\nQ: {row['question']}?")
            ids.append(tokenizer.encode(
                _render(tokenizer, [], user, bot_prefix="A:"),
                add_special_tokens=False))

    elif name == "mmlu":
        ds = _take(load_dataset("cais/mmlu", "all", split="test"), n, cfg)
        style = getattr(cfg, "mmlu_style", "simple_evals")
        if style == "5shot":
            dev = load_dataset("cais/mmlu", "all", split="dev")
            by_subject: Dict[str, list] = {}
            for row in dev:
                by_subject.setdefault(row["subject"], []).append(row)
            for row in ds:
                subj = row["subject"].replace("_", " ")
                hint = (f"There is a single choice question about {subj}. "
                        f"Answer the question by replying A, B, C or D.")
                shots = []
                for ex in by_subject.get(row["subject"], [])[:5]:
                    c = ex["choices"]
                    shots.append((
                        f"{hint}\nQuestion: {ex['question']}\nA. {c[0]}\n"
                        f"B. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer: ",
                        f"{'ABCD'[int(ex['answer'])]}\n"))
                c = row["choices"]
                user = (f"{hint}\nQuestion: {row['question']}\nA. {c[0]}\n"
                        f"B. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer: ")
                ids.append(tokenizer.encode(
                    _render(tokenizer, shots, user), add_special_tokens=False))
        else:
            for row in ds:
                c = row["choices"]
                user = _MMLU_SIMPLE_EVALS.format(
                    input=row["question"], A=c[0], B=c[1], C=c[2], D=c[3])
                ids.append(tokenizer.encode(
                    _render(tokenizer, [], user), add_special_tokens=False))

    else:
        raise ValueError(
            f"OpenCompass prompts not transcribed for dataset {name!r} "
            f"(have: gsm8k, mbpp, triviaqa, mmlu)")

    return ds, ids


# ── scorers matching the OpenCompass output formats ───────────────────────
# The repo's SCORERS assume its own prompt formats (```python blocks for MBPP,
# a bare letter for MMLU) and mis-score OpenCompass-formatted generations.

def _oc_gsm8k(gen_text: str, ref: dict) -> Tuple[bool, str]:
    """opencompass.datasets.gsm8k_postprocess: cut at the next 'Question:',
    take the last number."""
    m = re.search(r"####\s*([\-\d,\.]+)", ref.get("answer", ""))
    if not m:
        return False, "no_ref"
    gold = m.group(1).replace(",", "").rstrip(".")
    text = gen_text.split("Question:")[0]
    nums = re.findall(r"-?\d+\.\d+|-?\d+", text.replace(",", ""))
    if not nums:
        return False, f"no_num gold={gold}"
    try:
        return abs(float(nums[-1]) - float(gold)) < 1e-4, f"gold={gold} pred={nums[-1]}"
    except ValueError:
        return False, f"parse_fail gold={gold}"


def _oc_mbpp(gen_text: str, ref: dict) -> Tuple[bool, str]:
    """MBPPEvaluator: the answer sits between [BEGIN] and [DONE], usually
    wrapped in single quotes."""
    from speculative_decoding.Experiment_Backend.self_draft_compare import _exec_isolated
    text = gen_text
    if "[BEGIN]" in text:
        text = text.split("[BEGIN]", 1)[1]
    text = text.split("[DONE]", 1)[0].strip()
    # OpenCompass's MBPPEvaluator strips the wrapping quotes INDEPENDENTLY, which
    # matters: a single flipped token can drop the closing quote, and requiring a
    # matched pair would then leave a leading "'" and turn a correct program into
    # a SyntaxError. Scoring must not be more brittle than the reference harness.
    if text.startswith("'"):
        text = text[1:]
    if text.endswith("'"):
        text = text[:-1]
    code = text.replace("\\r", "\r").replace("\\n", "\n")
    tests = "\n".join(ref.get("test_list", []))
    return _exec_isolated(code + "\n" + tests + "\n")


def _oc_mmlu(gen_text: str, ref: dict) -> Tuple[bool, str]:
    """match_answer_pattern r'(?i)ANSWER\\s*:\\s*([A-D])', falling back to the
    repo scorer's looser search so 5-shot short answers still score."""
    raw = ref.get("answer")
    try:
        gold = "ABCD"[int(raw)]
    except (TypeError, ValueError, IndexError):
        return False, f"bad_ref={raw!r}"
    m = re.search(r"(?i)ANSWER\s*:\s*\(?\s*([A-D])", gen_text)
    if m:
        return m.group(1).upper() == gold, f"gold={gold} pred={m.group(1).upper()}"
    m = re.search(r"(?:^|[^A-Za-z])([ABCD])(?:[^A-Za-z]|$)", gen_text)
    if m:
        return m.group(1) == gold, f"gold={gold} pred={m.group(1)} (fallback)"
    return False, f"no_letter gold={gold}"


def _oc_triviaqa(gen_text: str, ref: dict) -> Tuple[bool, str]:
    """Same alias matching as the repo scorer; the OpenCompass prompt only
    changes the surface form ('The answer is X')."""
    from speculative_decoding.Experiment_Backend.self_draft_compare import score_triviaqa
    return score_triviaqa(gen_text, ref)


SCORERS_OC = {
    "gsm8k": _oc_gsm8k,
    "mbpp": _oc_mbpp,
    "mmlu": _oc_mmlu,
    "triviaqa": _oc_triviaqa,
}
