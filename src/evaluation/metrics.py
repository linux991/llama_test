"""Stylometric features for English and Russian horror/thriller texts."""

from __future__ import annotations

import re
import statistics
from collections import Counter


EN_FEAR_WORDS = {
    "afraid", "blood", "body", "cold", "dark", "darkness", "dead", "death",
    "fear", "ghost", "grave", "horror", "monster", "night", "nightmare",
    "scream", "shadow", "terror", "whisper",
}

EN_SUSPENSE_WORDS = {
    "behind", "breath", "door", "footstep", "hall", "listen", "quiet",
    "silence", "still", "wait", "watch", "window",
}

RU_FEAR_WORDS = {
    "боль", "гроб", "жуткий", "кровь", "мертвый", "могила", "ночь",
    "призрак", "страх", "страшный", "темнота", "тень", "труп", "ужас",
    "холод", "шепот",
}

RU_SUSPENSE_WORDS = {
    "дверь", "ждать", "коридор", "окно", "позади", "слушать", "тишина",
    "тихий", "шаг", "шорох",
}

EN_CLICHES = [
    "blood ran cold",
    "couldn't shake the feeling",
    "heart pounded",
    "heart hammered",
    "shiver down my spine",
    "the darkness seemed",
    "whispered my name",
]

RU_CLICHES = [
    "кровь застыла",
    "мурашки по коже",
    "сердце бешено",
    "сердце колотилось",
    "темнота словно",
    "шепнул мое имя",
    "шепнуло мое имя",
]


def words(text: str, language: str) -> list[str]:
    if language == "ru":
        return re.findall(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)?", str(text).lower())
    return re.findall(r"[A-Za-z][A-Za-z']*", str(text).lower())


def sentence_count(text: str) -> int:
    matches = re.findall(r"[^.!?…]+[.!?…]+", str(text))
    return max(1, len(matches))


def repetition_score(tokens: list[str]) -> float:
    """Return the proportion of repeated bigram occurrences.

    A higher value means that the text repeats the same two-word
    sequences more often.
    """

    if len(tokens) < 2:
        return 0.0

    bigrams = list(zip(tokens, tokens[1:]))
    counts = Counter(bigrams)

    repeated_occurrences = sum(
        count - 1
        for count in counts.values()
        if count > 1
    )

    return repeated_occurrences / len(bigrams)


def count_cliches(text: str, language: str) -> int:
    lowered = str(text).lower()
    patterns = RU_CLICHES if language == "ru" else EN_CLICHES
    return sum(lowered.count(pattern) for pattern in patterns)


def stylometric_features(text: str, language: str) -> dict[str, float]:
    token_list = words(text, language)
    fear_words = RU_FEAR_WORDS if language == "ru" else EN_FEAR_WORDS
    suspense_words = RU_SUSPENSE_WORDS if language == "ru" else EN_SUSPENSE_WORDS
    word_count = len(token_list)

    return {
        "char_count": float(len(str(text))),
        "word_count": float(word_count),
        "sentence_count": float(sentence_count(text)),
        "avg_word_length": statistics.fmean([len(word) for word in token_list]) if token_list else 0.0,
        "type_token_ratio": len(set(token_list)) / max(1, word_count),
        "question_mark_count": float(str(text).count("?")),
        "exclamation_mark_count": float(str(text).count("!")),
        "ellipsis_count": float(str(text).count("...") + str(text).count("…")),
        "fear_word_rate": sum(1 for word in token_list if word in fear_words) / max(1, word_count),
        "suspense_word_rate": sum(1 for word in token_list if word in suspense_words) / max(1, word_count),
        "cliche_count": float(count_cliches(text, language)),
        "repetition_score": repetition_score(token_list),
    }


FEATURE_COLUMNS = [
    "char_count",
    "word_count",
    "sentence_count",
    "avg_word_length",
    "type_token_ratio",
    "question_mark_count",
    "exclamation_mark_count",
    "ellipsis_count",
    "fear_word_rate",
    "suspense_word_rate",
    "cliche_count",
    "repetition_score",
]
