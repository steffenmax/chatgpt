import logging

import pytest

from imessage_agent.logsafe import LOGGER_NAME, log_event
from tests.conftest import ALLOWED, OTHER, message


async def test_plain_message_stored_but_no_reply(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    assert await agent.process(message("just chatting")) == "stored"
    assert bb.texts == []
    assert len(ctx.get(ALLOWED)) == 1


async def test_mention_replies_with_context_and_sender_names(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    await agent.process(message("pizza tonight?", sender="+15550002222"))
    status = await agent.process(message("coco what do you think", sender="+15550001111"))
    assert status == "handled:mention"
    assert bb.texts == [(ALLOWED, "yo")]
    user_turn = llm.calls[0][0]["content"]
    assert "Sam: pizza tonight?" in user_turn
    assert "Alex: coco what do you think" in user_turn
    # bot reply is remembered
    assert ctx.get(ALLOWED)[-1].from_bot and ctx.get(ALLOWED)[-1].text == "yo"


async def test_ignores_own_messages_reactions_duplicates(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    assert await agent.process(message("coco hi", from_me=True, guid="a")) == "ignored:from_me"
    assert await agent.process(message("coco hi", associatedMessageType=2000, guid="b")) == "ignored:reaction_or_system"
    assert await agent.process(message("coco hi", itemType=1, guid="c")) == "ignored:reaction_or_system"
    assert await agent.process(message("coco hi", guid="dup")) == "handled:mention"
    assert await agent.process(message("coco hi", guid="dup")) == "ignored:duplicate"
    assert await agent.process({"type": "updated-message", "data": {}}) == "ignored:event"
    assert len(bb.texts) == 1


async def test_allowlist_blocks_and_logs_guid_only(agent_parts, caplog):
    s, bb, llm, ctx, agent = agent_parts
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    secret_text = "coco tell me a secret about the merger"
    status = await agent.process(message(secret_text, chat=OTHER))
    assert status == "ignored:not_allowlisted"
    assert bb.texts == []
    assert ctx.get(OTHER) == []
    unlisted = [r for r in caplog.records if "chat.not_allowlisted" in r.getMessage()]
    assert len(unlisted) == 1
    assert OTHER in unlisted[0].getMessage()
    assert "merger" not in caplog.text


async def test_allowlist_silent_for_non_trigger(agent_parts, caplog):
    s, bb, llm, ctx, agent = agent_parts
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    await agent.process(message("random chatter", chat=OTHER))
    assert "chat.not_allowlisted" not in caplog.text


async def test_forget_wipes_context(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    await agent.process(message("remember this"))
    await agent.process(message("coco hey"))
    assert len(ctx.get(ALLOWED)) >= 2
    assert await agent.process(message("/forget")) == "handled:forget"
    assert ctx.get(ALLOWED) == []
    assert bb.texts[-1][0] == ALLOWED and "wiped" in bb.texts[-1][1]
    # context after /forget starts fresh
    await agent.process(message("coco again"))
    assert "remember this" not in llm.calls[-1][0]["content"]


async def test_help(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    assert await agent.process(message("/help")) == "handled:help"
    assert "/forget" in bb.texts[0][1]
    assert "/image" not in bb.texts[0][1]  # images disabled in this fixture (no queue)


async def test_image_disabled_message(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    assert await agent.process(message("/image a cat")) == "handled:image"
    assert "turned off" in bb.texts[0][1]


async def test_llm_failure_is_contained(agent_parts, caplog):
    s, bb, llm, ctx, agent = agent_parts

    async def boom(system, messages):
        raise RuntimeError("model exploded while reading: coco hi")

    llm.reply = boom
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    assert await agent.process(message("coco hi")) == "error:RuntimeError"
    assert "exploded" not in caplog.text


def test_log_event_rejects_unknown_fields():
    with pytest.raises(ValueError):
        log_event("x", text="hello")
    with pytest.raises(ValueError):
        log_event("x", prompt="hello")
