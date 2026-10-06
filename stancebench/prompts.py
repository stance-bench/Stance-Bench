"""Direct and per-comment reader prompts for frozen language models."""
from __future__ import annotations
from .data import candidate_order

SYSTEM_DIRECT = (
    "You are simulating one specific Hacker News user (the subject) from their comment history. "
    "Predict how they will respond in a new discussion.\nReturn JSON with:\n"
    "- reasoning: at most three sentences connecting their history to this prediction.\n"
    "- predicted_comment: the comment this subject would most plausibly write, in their own "
    "voice and typical length. Write the comment itself, never a description of it.\n"
    "- label_id: the candidate position that comment takes. Use OTHER if their position is not "
    "listed.\nThe comment and the label must express the same stance."
)

SYSTEM_READER = (
    "You infer a Hacker News user's likely position in a discussion from one comment they wrote elsewhere. "
    "Answer in JSON."
)

LETTERS = "ABCDEFGHIJKLMN"


def clip(text, limit):
    text = text or ""
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def direct_prompt(context):
    history = context["history"][-8:]
    if len(history) != 8:
        raise ValueError("Direct prediction requires eight annotated history entries")
    lines = ["== SUBJECT HISTORY: their 8 most recent labeled discussions, oldest first =="]
    for index, entry in enumerate(history, 1):
        days = round(float(entry.get("hours_before_target") or 0) / 24, 1)
        lines += [f"[{index}] Discussion: {entry.get('thread_title')}  ({days} days before)",
                  f"    Argued over: {entry.get('central_question')}",
                  f"    Position the subject took: {entry.get('position_taken')}"]
        if entry.get("positions_not_taken"):
            lines.append("    Positions they did not take: " + " | ".join(entry["positions_not_taken"]))
        if entry.get("user_comment"):
            lines.append(f'    Subject wrote: "{clip(entry["user_comment"], 500)}"')
    stimulus = context["target_stimulus"]
    lines += ["", "== NEW DISCUSSION the subject is about to comment in ==",
              f"Title: {stimulus.get('thread_title')}",
              "Link domain: withheld"]
    if stimulus.get("story_text"):
        lines.append("Story text:\n" + clip(stimulus["story_text"], 1200))
    else:
        lines.append("(The linked article body is not available; judge from the title and comments.)")
    visible = stimulus.get("visible_thread_comments") or []
    if visible:
        lines.append("Comments already in the thread (as a visitor would first see it):")
        lines += ["  - " + clip(comment.get("text"), 300) for comment in visible[:8]]
    lines += [f"The discussion's central question: {stimulus.get('central_question')}", "",
              "== CANDIDATE POSITIONS: the subject's comment will take exactly one "
              "(listed in no particular order) =="]
    labels = {option["id"]: option for option in stimulus["candidate_labels"]}
    for label in candidate_order(context):
        option = labels[label]
        row = f"[{label}] {option['name']}: {option.get('definition', '')}"
        if option.get("boundary"):
            row += f" (Not this if: {option['boundary']})"
        lines.append(row)
    return "\n".join(lines)


def reader_prompt(context, order, text):
    stimulus = context["target_stimulus"]
    labels = {option["id"]: option for option in stimulus["candidate_labels"]}
    lines = [f"Central question: {stimulus['central_question']}",
             f"Discussion: {stimulus['thread_title']}", "", "Positions:"]
    for index, label in enumerate(order):
        option = labels[label]
        value = f"{option.get('name') or label}: {option.get('definition') or ''}".strip()
        if option.get("boundary"):
            value += f" (Not this if: {option['boundary']})"
        lines.append(f"[{LETTERS[index]}] {value}")
    lines += ["[U] The comment says nothing about which of these positions this person holds.", "",
              "This person wrote the following comment earlier, elsewhere on Hacker News:",
              '"' + (text or "")[:600] + '"', "",
              "Which of the positions above is this person most likely to take in the discussion? "
              'Return JSON {"answer": "<letter>"}.']
    return "\n".join(lines)
