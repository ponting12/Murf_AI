import os
import sys
import pytest
from unittest.mock import MagicMock

# Mock the plugins BEFORE importing the agent module
# This prevents ImportError if the environment is missing specific plugins
sys.modules["livekit.plugins.murf"] = MagicMock()
sys.modules["livekit.plugins.silero"] = MagicMock()
sys.modules["livekit.plugins.deepgram"] = MagicMock()
sys.modules["livekit.plugins.noise_cancellation"] = MagicMock()
sys.modules["livekit.plugins.openai"] = MagicMock()
sys.modules["livekit.plugins.turn_detector"] = MagicMock()
sys.modules["livekit.plugins.turn_detector.multilingual"] = MagicMock()

from livekit.agents import RunContext

# Add the 'backend' directory to sys.path so we can import 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the WellnessAssistant and helper functions
from src.wellness_agent import WellnessAssistant, WELLNESS_LOG_PATH, _load_log, _save_log

@pytest.fixture
def assistant():
    """Create a fresh WellnessAssistant with a clean log file."""
    if os.path.exists(WELLNESS_LOG_PATH):
        os.remove(WELLNESS_LOG_PATH)
    return WellnessAssistant()

@pytest.mark.asyncio
async def test_add_checkin_persists(assistant: WellnessAssistant):
    ctx = MagicMock()
    result = await assistant.add_checkin(
        ctx,
        mood="happy",
        energy="high",
        stress="none",
        objectives="write report, walk",
    )
    assert "saved" in result.lower()
    entries = _load_log()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["mood"] == "happy"
    assert entry["energy"] == "high"
    assert entry["stress"] == "none"
    assert entry["objectives"] == ["write report", "walk"]

@pytest.mark.asyncio
async def test_get_last_checkin_returns_summary(assistant: WellnessAssistant):
    ctx = MagicMock()
    await assistant.add_checkin(
        ctx,
        mood="tired",
        energy="low",
        stress="work",
        objectives="rest",
    )
    summary = await assistant.get_last_checkin(ctx)
    assert "tired" in summary.lower()
    assert "low" in summary.lower()
