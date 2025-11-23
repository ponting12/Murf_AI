import logging
import json

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
from livekit.plugins import murf, silero, deepgram, noise_cancellation, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel

# Load environment variables
load_dotenv(".env.local")

logger = logging.getLogger("agent")

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly barista at CodeBrew Coffee.
Your goal is to take the customer's order efficiently and warmly.
You need to collect the following information to complete an order:
- Drink Type (e.g., Latte, Cappuccino, Americano, Espresso)
- Size (Small, Medium, Large)
- Milk Type (Whole, Skim, Oat, Almond, Soy, None)
- Extras (e.g., Vanilla Syrup, Extra Shot, Whipped Cream, None)
- Customer Name

Current Order State is available to you. Ask clarifying questions to fill in any missing details (null values).
If the user specifies multiple things at once, update them all.
For 'extras', if the user adds something, append it to the list.
Once you have all the details (Drink Type, Size, Milk, Name), recite the full order back to the customer for confirmation.
If they confirm, use the 'finalize_order' tool to save the order.
If they want to change something, use 'update_order'.

Be conversational, friendly, and helpful.
""",
        )
        self.order_state = {
            "drinkType": None,
            "size": None,
            "milk": None,
            "extras": [],
            "name": None,
        }

    @function_tool
    async def update_order(
        self,
        ctx: RunContext,
        drinkType: str = None,
        size: str = None,
        milk: str = None,
        extras: str = None,
        name: str = None,
    ):
        """Update the order details based on user input."""
        if drinkType:
            self.order_state["drinkType"] = drinkType
        if size:
            self.order_state["size"] = size
        if milk:
            self.order_state["milk"] = milk
        if name:
            self.order_state["name"] = name
        if extras:
            new_extras = [e.strip() for e in extras.split(',') if e.strip()]
            for extra in new_extras:
                if extra not in self.order_state["extras"]:
                    self.order_state["extras"].append(extra)
        return f"Order updated. Current state: {json.dumps(self.order_state)}"

    @function_tool
    async def finalize_order(self, ctx: RunContext):
        """Finalize and save the order after confirmation."""
        with open("order.json", "w") as f:
            json.dump(self.order_state, f, indent=2)
        return "Order finalized and saved to order.json."

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

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session with the Assistant agent
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Connect to the room and begin interaction
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
