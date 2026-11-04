"""Shared helpers for building the new paired datasets (HC3 and PERSUADE-based)."""
import random
import re

# Tokenisation artefacts (" ," " ." "( " ...) appear in some human-written sources (e.g. HC3
# Reddit/WikiQA answers) but never in LLM output. Left in place they would let a detector
# separate the classes on formatting alone, so the same normaliser is applied to BOTH sides.
_SPACE_FIXES = [(r"\s+([,.;:!?%)\]])", r"\1"), (r"([(\[])\s+", r"\1"), (r"\s+n't\b", "n't"),
                (r"\s+'(s|re|ve|ll|d|m)\b", r"'\1"), (r"``\s*|''", '"')]


def normalise(text):
    text = " ".join(str(text).split())
    for pat, rep in _SPACE_FIXES:
        text = re.sub(pat, rep, text)
    return text.strip()


def trim_pair(human, machine, max_words=250):
    """Cut both texts to the same number of words (<= max_words) so length is not a cue.
    This mirrors trim_to_shorter_length() in run.py, which the original paper applies."""
    hw, mw = human.split(), machine.split()
    n = min(len(hw), len(mw), max_words)
    return " ".join(hw[:n]), " ".join(mw[:n]), n


def interleave(groups, seed=0):
    """groups: dict name -> list of rows. Shuffle each group, then interleave round-robin so
    that the first n rows of the output cover every group (run.py takes the first n_samples)."""
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    out, longest = [], max((len(g) for g in groups.values()), default=0)
    for i in range(longest):
        for name in sorted(groups):
            if i < len(groups[name]):
                out.append(groups[name][i])
    return out
