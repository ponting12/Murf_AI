import logging
import json
import os
from datetime import datetime

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, deepgram, openai
import livekit.plugins.noise_cancellation as noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env.local")
load_dotenv(env_path)

logger = logging.getLogger("wellness_agent")

WELLNESS_LOG_PATH = os.path.join(os.path.dirname(__file__), "wellness_log.json")

# Ensure required LiveKit environment variables are set; provide defaults for development.
required_vars = ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"]
dummy_detected = False
for var in required_vars:
    if var not in os.environ or os.environ.get(var, "").startswith("dummy_"):
        dummy_detected = True
        logger.error(f"Environment variable {var} is missing or set to dummy value.")
if dummy_detected:
    logger.error("LIVEKIT environment variables are not properly configured. Please set LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET in your .env.local or environment.")
    # Exit the process to avoid silent failures
    import sys; sys.exit(1)
def _load_log():
    """Load the wellness log JSON file, returning a list of entries."""
    if not os.path.exists(WELLNESS_LOG_PATH):
        return []
    try:
        with open(WELLNESS_LOG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load wellness log: {e}")
        return []


def _save_log(entries):
    """Write the list of entries back to the JSON file with pretty formatting."""
    try:
        with open(WELLNESS_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save wellness log: {e}")


class WellnessAssistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""
You are a friendly, supportive health & wellness voice companion. Your job is to conduct a brief daily check‑in with the user.

The flow you should follow (you can vary wording, but keep the structure):
1. Greet the user.
2. Ask about their mood (free‑text) and energy level.
3. Ask if anything is currently stressing them.
4. Ask the user to share 1‑3 practical objectives they would like to accomplish today (work, self‑care, exercise, etc.).
5. Summarise the information you gathered in a short, encouraging sentence.
6. Persist the check‑in using the `add_checkin` function tool.
7. Refer back to the previous day’s entry (if any) with a gentle, supportive comment, e.g., "Last time you mentioned low energy; how does today feel?".
8. Close with a brief recap and ask for confirmation.

**Never** give medical advice, diagnose, or make any health claims. Keep suggestions small, actionable, and non‑clinical (e.g., "consider a short walk", "break a big task into smaller steps").
"""
        )
        # Load any existing log so we can reference the last entry.
        self.last_entry = None
        entries = _load_log()
        if entries:
            self.last_entry = entries[-1]

    @function_tool
    async def add_checkin(
        self,
        ctx: RunContext,
        mood: str,
        energy: str,
        stress: str,
        objectives: str,
    ) -> str:
        """Persist a daily wellness check‑in.
        Parameters are free‑text strings supplied by the LLM after the conversation.
        The function returns a short confirmation message.
        """
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mood": mood,
            "energy": energy,
            "stress": stress,
            "objectives": [obj.strip() for obj in objectives.split(',') if obj.strip()],
        }
        entries = _load_log()
        entries.append(entry)
        _save_log(entries)
        self.last_entry = entry
        return "Check‑in saved successfully."

    @function_tool
    async def get_last_checkin(self, ctx: RunContext) -> str:
        """Return a brief summary of the previous day's check‑in, or an empty string if none exists."""
        if not self.last_entry:
            return ""
        mood = self.last_entry.get("mood", "")
        energy = self.last_entry.get("energy", "")
        return f"Yesterday you felt {mood.lower()} with {energy.lower()} energy."


def prewarm(proc: JobProcess):
    # Load VAD model once for all workers
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Add room name to log context for easier debugging
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=openai.LLM(
            model="gpt-4o-mini",
            temperature=0.7,
        ),
        tts=murf.TTS(
            voice="en-US-matthew",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection (same as barista example)
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=WellnessAssistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    # If using dummy LiveKit env vars, skip launching the LiveKit app.
    dummy_vars = [var for var in [os.getenv("LIVEKIT_URL"), os.getenv("LIVEKIT_API_KEY"), os.getenv("LIVEKIT_API_SECRET")] if var and var.startswith("dummy_")]
    if dummy_vars:
        print("Running in development mode with dummy LiveKit configuration. Agent not started.")
    else:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
