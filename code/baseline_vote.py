"""Majority-vote baseline using the official numeric parser."""

from collections import Counter

from common import parse_answer


def predict(bank, arrays, checkpoint):
    choices = []

    for row in bank:
        candidates = row["candidates"]
        answers = [
            parse_answer(candidate["text"])
            for candidate in candidates
        ]

        # Unparseable answers do not participate in voting.
        counts = Counter(
            answer for answer in answers
            if answer is not None
        )

        if not counts:
            # Fixed fallback when no candidate has a parseable answer.
            choices.append(0)
            continue

        highest_count = max(counts.values())

        # Break ties by selecting the earliest candidate.
        choice = next(
            candidate["index"]
            for candidate, answer in zip(candidates, answers)
            if answer is not None and counts[answer] == highest_count
        )

        choices.append(int(choice))

    return choices