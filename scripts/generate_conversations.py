"""
Generate synthetic multi-topic conversations for ground-truth testing.
=====================================================================

Creates transcripts with **known topic boundaries** so we can measure
the precision / recall of the boundary detector and Kalman filter.

Each "block" is a paragraph of coherent text on a single topic.
Transitions between blocks are the ground-truth boundaries.

Usage::

    python scripts/generate_conversations.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ── Add src to path for direct execution ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Synthetic conversations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TOPIC_BLOCKS = {
    "topic_shift_001": [
        {
            "topic": "dogs",
            "text": (
                "I've always loved dogs. Golden retrievers are my favorite breed "
                "because of their gentle temperament and loyalty. When I was growing "
                "up we had a golden named Biscuit who would follow me everywhere. "
                "Dogs really are remarkable companions. They can sense your emotions "
                "and respond with genuine affection. Training a puppy takes patience "
                "but the bond you build is incredibly rewarding. Walking a dog every "
                "morning also keeps you healthy and connected to your neighborhood."
            ),
        },
        {
            "topic": "airplanes",
            "text": (
                "Speaking of travel, I recently took a flight to Tokyo. The Boeing "
                "787 Dreamliner is an incredible piece of engineering. The composite "
                "fuselage allows for higher cabin pressure and larger windows. "
                "Modern jet engines are remarkably efficient compared to earlier "
                "turbofans. The physics of flight still amazes me — lift generated "
                "by differential pressure over the wing surface. Air traffic control "
                "coordinates thousands of flights simultaneously using radar and "
                "satellite tracking systems."
            ),
        },
        {
            "topic": "dogs_return",
            "text": (
                "Anyway, back to dogs. My neighbor just adopted a rescue greyhound. "
                "Former racing dogs make surprisingly calm house pets. They're called "
                "forty-five mile per hour couch potatoes because they sprint then "
                "sleep all day. Dog adoption has become much more popular in recent "
                "years. Shelters and rescue organizations do amazing work matching "
                "dogs with families. Every dog deserves a loving home."
            ),
        },
        {
            "topic": "cooking",
            "text": (
                "Last night I tried making homemade pasta for the first time. You "
                "just need flour, eggs, and a bit of olive oil. Kneading the dough "
                "is meditative — you can feel the gluten developing as it becomes "
                "smooth and elastic. Rolling it thin enough is the hardest part "
                "without a pasta machine. I made a simple aglio e olio with garlic, "
                "chili flakes, and parsley. The key is to cook the garlic slowly so "
                "it infuses the oil without burning. Fresh pasta cooks in just two "
                "minutes compared to ten for dried."
            ),
        },
        {
            "topic": "mathematics",
            "text": (
                "I've been reading about topology lately. The idea that a coffee cup "
                "and a donut are topologically equivalent blows my mind. Continuous "
                "deformations preserve the fundamental structure — one hole. Euler's "
                "formula connects the vertices, edges, and faces of polyhedra in an "
                "elegant relationship. Mathematics has this beautiful quality where "
                "abstract structures reveal deep truths about reality. Category "
                "theory takes this even further by studying the relationships between "
                "mathematical structures themselves."
            ),
        },
    ],
    "fixation_test": [
        {
            "topic": "weather",
            "text": (
                "The weather today is partly cloudy with a chance of rain. "
                "The temperature is around sixty-five degrees. "
                "It might rain later this afternoon. "
                "The forecast says partly cloudy. "
                "Temperatures around sixty-five. "
                "Clouds moving in from the west. "
                "Still about sixty-five degrees outside. "
                "The weather hasn't changed much. "
                "Still partly cloudy out there. "
                "Temperature holding steady at sixty-five."
            ),
        },
    ],
    "rapid_shifts": [
        {
            "topic": "ocean",
            "text": "The ocean is vast and deep. Marine biology studies incredible ecosystems in coral reefs.",
        },
        {
            "topic": "space",
            "text": "NASA launched another mission to Mars. The rover will search for signs of ancient microbial life.",
        },
        {
            "topic": "music",
            "text": "Jazz improvisation requires deep musical knowledge. Coltrane changed saxophone forever.",
        },
        {
            "topic": "ocean_return",
            "text": "Whales migrate thousands of miles through the ocean. Their songs can travel enormous distances underwater.",
        },
        {
            "topic": "politics",
            "text": "Congressional debates shape national policy. Bipartisan cooperation remains challenging in polarized times.",
        },
        {
            "topic": "space_return",
            "text": "The James Webb telescope revealed galaxies from the early universe. Dark matter remains one of physics's great mysteries.",
        },
    ],
}


def generate_all(out_dir: str | Path = "data/synthetic") -> None:
    """Generate all synthetic conversations and their ground-truth manifests."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, blocks in TOPIC_BLOCKS.items():
        # ── Write plain-text transcript ───────────────────────────────
        full_text = "\n\n".join(b["text"] for b in blocks)
        txt_path = out_dir / f"{name}.txt"
        txt_path.write_text(full_text, encoding="utf-8")

        # ── Write ground-truth manifest (JSON) ────────────────────────
        # Records character offsets of each block so we can later map
        # window indices to known topic boundaries.
        manifest: list[dict] = []
        offset = 0
        for b in blocks:
            length = len(b["text"])
            manifest.append({
                "topic": b["topic"],
                "char_start": offset,
                "char_end": offset + length,
                "length_chars": length,
            })
            offset += length + 2  # +2 for "\n\n" separator

        json_path = out_dir / f"{name}_ground_truth.json"
        json_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        print(f"✓ Generated {txt_path}  ({len(blocks)} blocks, {len(full_text)} chars)")
        print(f"  Ground truth: {json_path}")


if __name__ == "__main__":
    generate_all()
