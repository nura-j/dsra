"""Top-K thresholds as shares of the token x layer grid (descriptive only).
The GPT-2 tokenizer is used: GPT-J-6B shares its BPE vocabulary for ordinary
text, the length statistics are the same.
"""

import statistics as st

GPTJ_LAYERS = 28
ANALYSIS_TEMPLATE = " is often referred to as:"


def build_pool(data, tokenizer, variant):
    """First definition per alphabetic, single-token definiendum.

    Returns ``(pre, pool)``: all such words, and those with <= 25-word definitions.
    """

    def single_token(word):
        text = word if variant == "bare" else " " + word
        return len(tokenizer.encode(text)) == 1

    firsts = {}
    for d in data:
        if d["word"].isalpha() and single_token(d["word"]):
            firsts.setdefault(d["word"], d)
    pre = list(firsts.values())
    pool = [d for d in pre if len(d["definition"].split()) <= 25]
    return pre, pool


def grid_stats(data, tokenizer, layers=GPTJ_LAYERS, template=ANALYSIS_TEMPLATE,
               variants=("space", "bare")):
    """Grid-share statistics for each single-token variant."""
    out = {}
    for variant in variants:
        pre, pool = build_pool(data, tokenizer, variant)
        lens = [len(tokenizer.encode(d["definition"].title() + template)) for d in pool]
        grids = [n * layers for n in lens]
        s10 = [10 / g * 100 for g in grids]
        s50 = [50 / g * 100 for g in grids]
        lens_sorted = sorted(lens)
        out[variant] = {
            "n_pool": len(pool),
            "coverage_le25_words_pct": round(len(pool) / len(pre) * 100, 1),
            "prompt_tokens_mean": round(st.mean(lens), 1),
            "prompt_tokens_median": st.median(lens),
            "prompt_tokens_p10": lens_sorted[int(0.1 * len(lens))],
            "prompt_tokens_p90": lens_sorted[int(0.9 * len(lens))],
            "grid_cells_mean": round(st.mean(grids)),
            "grid_cells_median": st.median(grids),
            "top10_grid_share_pct_mean": round(st.mean(s10), 2),
            "top10_grid_share_pct_median": round(st.median(s10), 2),
            "top10_grid_share_pct_sd": round(st.pstdev(s10), 2),
            "top50_grid_share_pct_mean": round(st.mean(s50), 2),
            "top50_grid_share_pct_median": round(st.median(s50), 2),
            "top50_grid_share_pct_sd": round(st.pstdev(s50), 2),
        }
    return out
