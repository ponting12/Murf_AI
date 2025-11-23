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
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")


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
            "name": None
        }

    @function_tool
    async def update_order(
        self, 
        ctx: RunContext, 
        drinkType: str = None, 
        size: str = None, 
        milk: str = None, 
        extras: str = None, 
        name: str = None
    ):
        """
        Update the order details. 
        Use this tool when the user provides information about their order.
        args:
            drinkType: The type of coffee/drink.
            size: The size of the drink.
            milk: The type of milk.
            extras: Any extra additions (e.g., "Vanilla Syrup"). If multiple, call this tool multiple times or pass a comma-separated string.
            name: The customer's name.
        """
        if drinkType:
            self.order_state["drinkType"] = drinkType
        if size:
            self.order_state["size"] = size
        if milk:
            self.order_state["milk"] = milk
        if name:
            self.order_state["name"] = name
        if extras:
            # Split by comma if multiple extras are provided in one string
            new_extras = [e.strip() for e in extras.split(',') if e.strip()]
            for extra in new_extras:
                if extra not in self.order_state["extras"]:
                    self.order_state["extras"].append(extra)
        
        return f"Order updated. Current state: {json.dumps(self.order_state)}"

    @function_tool
    async def finalize_order(self, ctx: RunContext):
        """
        Finalize and save the order after the user confirms it.
        """
        with open("order.json", "w") as f:
            json.dump(self.order_state, f, indent=2)
        
        return "Order finalized and saved to order.json."


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=deepgram.STT(model="nova-3"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=google.LLM(
                model="gemini-2.5-flash",
            ),
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=murf.TTS(
                voice="en-US-matthew", 
                style="Conversation",
                tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
                text_pacing=True
            ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # Metrics collection, to measure pipeline performance
    # For more information, see https://docs.livekit.io/agents/build/metrics/
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
