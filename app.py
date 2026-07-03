"""
Reel Hook and Caption Generator for Sacred Love Initiation.

Transcribes an uploaded video or audio file and generates an on-brand hook
and caption for Kate Joyner's Instagram reels using OpenAI Whisper and Claude.

Inputs required:
- OPENAI_API_KEY (Whisper transcription)
- ANTHROPIC_API_KEY (Claude generation)
- Set these in Streamlit Cloud secrets, or in a local .env file for development.

Process:
1. User uploads a video/audio file or pastes a transcript manually.
2. File is sent to OpenAI Whisper API for transcription (if uploaded).
3. Optional offering/CTA context is added to the prompt.
4. Transcript is sent to Claude with the full brand voice system prompt.
5. Hook (Title Case) and caption (with hashtags) are displayed with copy buttons.

Quality criteria:
- Hook is in Title Case, 5-12 words, curiosity-driven, answer not given in hook.
- Caption is 3-5 paragraphs in Kate's brand voice with no em dashes.
- Caption ends with exactly 5 hashtags led by #SacredLoveInitiation #KateJoyner.

Edge cases:
- File over 25MB: Whisper API rejects it. User is shown a clear error and offered
  the manual transcript field as a fallback.
- Unsupported format: Streamlit uploader blocks it at UI level.
- No speech detected: Claude receives the empty transcript and flags it gracefully.
- Offering field empty: Claude generates a general discovery CTA pointing to link in bio.

Failure handling:
- Whisper failure: Error displayed, manual transcript field offered as fallback.
- Claude failure: Error displayed, user can retry without re-uploading.
- Missing API keys: Startup check blocks the app and names the missing key(s).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import anthropic
import openai
import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

if load_dotenv is not None:
    load_dotenv()

SYSTEM_PROMPT = """You are a content assistant for Sacred Love Initiation, a love mystery school led by Kate Joyner, known as The Sacred Love Witch. Your task: read a reel transcript and output a hook and caption in Kate's brand voice.

## Who Kate Is

Kate Joyner is The Sacred Love Witch. A Sacred Love guide and High Priestess with over 20 years of devoted study in Soul Initiation, the Tantric Arts of the Deep Feminine, and Sacred Intimacy. She initiates people in deeper, higher, greater love, resurrecting love lives that feel dead, emptied, or battle-worn.

Sacred Love Initiation is a love mystery school. The mission: bring the feminine and masculine together, in all their light and dark, for renewed relationships that create beautifully supportive families.

Core audience: spiritually-aware men and women, 28-60s, who want deep sacred love and have tried conventional approaches without success.
Wounded beliefs to speak to: "I'm too much" (feminine) / "I'm not enough" (masculine).

## Brand Voice

Soft. Hungry. Ferocious. Honest. Hold all four simultaneously.
- Soft enough to invite
- Hungry enough to be real
- Ferocious enough to challenge
- Honest enough to be trusted

Balance: 50% mystical and evocative, 50% grounded and accessible. Never so esoteric that newcomers feel excluded. Never so grounded that the magic disappears.

## Brand Lexicon

Use these words and phrases naturally: Sacred Love, Sacred Union, Sacred Lovers, Sacred Partnership, Love Witch, Love Magic, Battle-worn, Heart-hungry, Resurrect, Resurrection, Invisible Shields, Invisible Swords, Underworld, Unearth, Reborn, The Sacred Lover, The Chalice, Sovereign, The dance (between masculine and feminine), Initiation, Devotion, Erotic, Eros.

## Voice Do's

- Evocative, sensual language that creates feeling
- Ground mystical concepts in relatable human experience
- Speak directly to one person, not "people in general"
- Let Kate's emotional voice through: "It saddens me," "I used to say that too"
- Weave CTAs in naturally as extension of the message, not a tonal shift
- Use "we" and "I" when referring to offerings

## Voice Don'ts

- No em dashes. Use commas or full stops instead.
- No staccato AI-sounding fragments ("Inherited. Absorbed. Shaped by.")
- No coach-speak ("step into," "activate," "unlock," "embody your truth")
- No sales-y language ("sign up now," "don't miss out," "limited spots")
- No bullet-point feature lists for programmes. Convey the felt experience instead.
- Never blame one gender
- No "This isn't about X. This is about Y." pattern. Reads as AI/sales copy.

## Hook Rules

Output: one line, Title Case (every word capitalised), 5-12 words maximum.

Psychology:
- Create curiosity. Do not complete the thought in the hook.
- Flow over fragments. The hook should feel like the start of a real sentence.
- Direct address outperforms observation. "If You Feel Like You're Too Much" beats "Women Often Feel Too Much."
- Ellipsis (...) creates pull and invites completion.
- Myth-busting wins in this niche. Challenge the dominant conversation in spiritual/relationship spaces.
- Kate's lived experience as a hook outperforms educational openers for high-ticket content.

Types that perform best:
- Myth-busting: "Being Feminine Isn't Being Soft All The Time"
- Pattern interrupt: "The Reason You Keep Attracting The Same Relationship"
- Direct address: "Women, If Your Love Still Feels Surface Level..."
- Curiosity gap: "Men Who Love Witches..."
- Authority/story: "I Spent A Decade In Devotional Celibacy. Here's What That Taught Me."
- Even if: "Even If You've Done All The Work And Still Can't Find Love"

## Caption Rules

- 3-5 short paragraphs. Let them breathe.
- Flow naturally from the hook. Do not repeat it.
- No staccato fragments. Use ellipsis (...) for trailing thoughts instead.
- Weave in the offering naturally as extension of the message. Not a tonal shift.
- CTA is one sentence. Tagline/essence language, not feature lists.
- End with a single line of exactly 5 hashtags. Always lead with #SacredLoveInitiation #KateJoyner then 3 content-relevant hashtags.

## Performing Examples

HOOK: A Woman Craves Two Things From A Man
CAPTION:
With your presence you create safety, containment, the feeling that she is held.

But she also craves your darkness.

These two forces seem contradictory but when you master both you master the art of deep, ecstatic love.

That is what Man of God teaches you. Ancient tantric shamanic wisdom you will not find just anywhere on the internet or learn from other guys.

Five weeks to go deep and deeper. Link in bio.

#ManOfGod #SacredMasculine #MasculineEmbodiment #SacredLove #ConsciousLove

---

HOOK: Men's Hearts Are The Most Sensitive
CAPTION:
For a man to come home to his heart without losing his masculinity is one of the most revelatory journeys he can take.

And when he does, everything opens. In himself, in his woman, in the dance between them.

That is what Man of God is here for. We start on Thursday. Link in bio.

#ManOfGod #SacredMasculine #MasculineEmbodiment #SacredLove #ConsciousLove

---

HOOK: Men Who Love Witches...
CAPTION:
Men who can meet witches are a rare and dying breed.

They have cultivated the capacity to meet the full force of the feminine without losing themselves in it. That is not accidental. That is initiation.

Man of God is where you can learn how. Link in bio.

#ManOfGod #SacredMasculine #MenWhoLoveWitches #FeminineInitiation #SacredLove

---

HOOK: Sacred Union Is Based On These Two Qualities
CAPTION:
A Sacred Union isn't a "normal" relationship. It is a devotional crucible for your deeper spiritual growth.

Ready to go deeper.

She & He, our nine week deep dive into love as a sacred living practice, is on pre-sale. DM me for more details...

#SacredLoveInitiation #KateJoyner #SacredUnion #DeepLove #SacredLove

---

HOOK: Women, If Your Love Still Feels Surface Level, This Is Probably Why

## Output Format

Output ONLY the following. No explanation, no preamble, no commentary.

HOOK: [hook in Title Case]

CAPTION:
[caption text]

#SacredLoveInitiation #KateJoyner #Hashtag3 #Hashtag4 #Hashtag5"""


def get_secret(key: str) -> str | None:
    """Read from Streamlit secrets (cloud) or environment (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.getenv(key)


def check_api_keys() -> list[str]:
    missing = []
    if not get_secret("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if not get_secret("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    return missing


def transcribe_file(file_bytes: bytes, filename: str) -> str:
    client = openai.OpenAI(api_key=get_secret("OPENAI_API_KEY"))
    suffix = Path(filename).suffix or ".mp4"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(model="whisper-1", file=f)
        return result.text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def generate_hook_caption(transcript: str, offering: str = "") -> str:
    client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
    user_message = f"Here is the reel transcript:\n\n{transcript}"
    if offering.strip():
        user_message += f"\n\nOffering / CTA to weave in naturally: {offering.strip()}"
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def parse_output(output: str) -> tuple[str, str]:
    hook = ""
    caption_lines = []
    in_caption = False
    for line in output.strip().split("\n"):
        upper = line.upper().strip()
        if upper.startswith("HOOK:"):
            hook = line.split(":", 1)[1].strip()
            in_caption = False
        elif upper.startswith("CAPTION:"):
            in_caption = True
            remainder = line.split(":", 1)[1].strip()
            if remainder:
                caption_lines.append(remainder)
        elif in_caption:
            caption_lines.append(line)
    caption = "\n".join(caption_lines).strip()
    return hook, caption


# ---- UI ----

st.set_page_config(
    page_title="Reel Generator | Sacred Love Initiation",
    page_icon="🌹",
    layout="centered",
)

st.title("Reel Hook & Caption Generator")
st.caption("Sacred Love Initiation with Kate Joyner")

missing_keys = check_api_keys()
if missing_keys:
    st.error(f"Missing API keys: {', '.join(missing_keys)}. Add them to Streamlit secrets.")
    st.stop()

st.divider()

col1, col2 = st.columns([3, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload reel video or audio",
        type=["mp4", "m4a", "mp3", "wav", "webm", "mpeg", "mpga"],
        help="Max 25MB. Supported: mp4, m4a, mp3, wav, webm.",
    )

with col2:
    offering = st.text_area(
        "Offering / CTA (optional)",
        placeholder="e.g. Man of God, 5 weeks, link in bio\nor: She & He, DM for details",
        height=110,
        help="What to mention in the caption CTA. Leave blank for a general discovery CTA.",
    )

st.markdown("**Or paste transcript manually:**")
manual_transcript = st.text_area(
    "manual_transcript",
    placeholder="Paste transcript here if you don't have the video file...",
    height=140,
    label_visibility="collapsed",
)

generate_btn = st.button(
    "Generate Hook & Caption", type="primary", use_container_width=True
)

if generate_btn:
    transcript = ""

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        size_mb = len(file_bytes) / (1024 * 1024)
        if size_mb > 25:
            st.error(
                f"File is {size_mb:.1f}MB. Whisper API limit is 25MB. "
                "Compress the video first, or paste the transcript manually."
            )
            st.stop()
        with st.spinner("Transcribing..."):
            try:
                transcript = transcribe_file(file_bytes, uploaded_file.name)
            except openai.OpenAIError as e:
                st.error(f"Transcription failed: {e}")
                st.info("Paste the transcript manually in the field above and try again.")
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error during transcription: {e}")
                st.stop()
    elif manual_transcript.strip():
        transcript = manual_transcript.strip()
    else:
        st.warning("Upload a video or paste a transcript to get started.")
        st.stop()

    with st.expander("Transcript", expanded=False):
        st.text(transcript)

    with st.spinner("Generating..."):
        try:
            raw_output = generate_hook_caption(transcript, offering)
        except anthropic.APIError as e:
            st.error(f"Caption generation failed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error during generation: {e}")
            st.stop()

    hook, caption = parse_output(raw_output)

    if not hook and not caption:
        st.warning("Could not parse the output. Raw response shown below.")
        st.text(raw_output)
        st.stop()

    st.divider()
    st.subheader("Hook")
    st.code(hook, language=None)
    st.subheader("Caption")
    st.code(caption, language=None)
