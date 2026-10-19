import asyncio
import logging

import pytest

from imessage_agent.handler import Agent, SenderResolver
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


async def test_quiet_question_answered_when_nobody_replies(clock):
    from imessage_agent.context import ContextStore
    from tests.conftest import FakeBB, FakeLLM, make_settings

    s = make_settings(quiet_question_seconds=0.05)
    bb, llm = FakeBB(), FakeLLM("sure")
    ctx = ContextStore(5, 3600, clock=clock)
    agent = Agent(s, bb, llm, ctx, images=None, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    assert await agent.process(message("anyone know when the game starts?", guid="q1")) == "stored"
    assert llm.calls == [] and bb.texts == []
    await asyncio.sleep(0.12)
    await agent.drain()
    assert len(llm.calls) == 1
    assert bb.texts == [(ALLOWED, "sure")]


async def test_quiet_question_dropped_when_someone_answers(clock):
    from imessage_agent.context import ContextStore
    from tests.conftest import FakeBB, FakeLLM, make_settings

    s = make_settings(quiet_question_seconds=0.05)
    bb, llm = FakeBB(), FakeLLM("sure")
    ctx = ContextStore(5, 3600, clock=clock)
    agent = Agent(s, bb, llm, ctx, images=None, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    await agent.process(message("anyone know when the game starts?", guid="q2"))
    await agent.process(message("8pm", sender="+15550002222", guid="a2"))
    await asyncio.sleep(0.12)
    await agent.drain()
    assert llm.calls == [] and bb.texts == []


async def test_quiet_question_off_by_default(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    assert s.quiet_question_seconds == 0
    await agent.process(message("is anyone around?", guid="q3"))
    await asyncio.sleep(0.05)
    await agent.drain()
    assert llm.calls == [] and bb.texts == []


def _cloud_agent(local_reply, clock):
    from imessage_agent.context import ContextStore
    from tests.conftest import FakeBB, FakeLLM, make_settings

    s = make_settings()
    bb, local, cloud = FakeBB(), FakeLLM(local_reply), FakeLLM("from the cloud")
    cloud.name = "anthropic"
    ctx = ContextStore(5, 3600, clock=clock)
    agent = Agent(
        s, bb, local, ctx, images=None,
        resolver=SenderResolver(None, {}, use_contacts=False),  # type: ignore[arg-type]
        cloud=cloud,
    )
    return agent, bb, local, cloud


async def test_mention_routed_to_cloud_when_router_says_yes(clock):
    agent, bb, local, cloud = _cloud_agent("yes", clock)
    assert await agent.process(message("coco who won last night?", guid="c1")) == "handled:mention"
    assert len(local.calls) == 1, "local model is asked only the yes/no question"
    assert local.calls[0][0]["content"].endswith("yes or no.")
    assert len(cloud.calls) == 1
    assert len(bb.texts) == 2, "a 'looking it up' notice, then the answer"
    assert "look" in bb.texts[0][1] or "check" in bb.texts[0][1] or "find out" in bb.texts[0][1]
    assert bb.texts[1] == (ALLOWED, "from the cloud")


async def test_mention_stays_local_when_router_says_no(clock):
    agent, bb, local, cloud = _cloud_agent("no", clock)
    await agent.process(message("coco roast me", guid="c2"))
    assert len(local.calls) == 2, "router call + the actual reply"
    assert cloud.calls == []
    assert bb.texts == [(ALLOWED, "no")]


async def test_ask_command_always_goes_to_cloud(clock):
    agent, bb, local, cloud = _cloud_agent("no", clock)
    assert await agent.process(message("/ask what time is kickoff", guid="c3")) == "handled:ask"
    assert local.calls == []
    assert len(cloud.calls) == 1
    # the question itself (without the slash command) is what the cloud model sees
    assert cloud.calls[0][-1]["content"].endswith("what time is kickoff")
    assert len(bb.texts) == 2 and bb.texts[1] == (ALLOWED, "from the cloud")


async def test_ask_without_cloud_falls_back_to_local(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    assert agent.cloud is None
    assert await agent.process(message("/ask something", guid="c4")) == "handled:ask"
    assert len(llm.calls) == 1 and bb.texts == [(ALLOWED, "yo")]


async def test_help_mentions_ask_only_with_cloud(clock, agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    await agent.process(message("/help", guid="h1"))
    assert "/ask" not in bb.texts[-1][1] and "nothing leaves it" in bb.texts[-1][1]
    agent2, bb2, _, _ = _cloud_agent("no", clock)
    await agent2.process(message("/help", guid="h2"))
    assert "/ask" in bb2.texts[-1][1] and "leaves the mac" in bb2.texts[-1][1]


async def test_cloud_failure_falls_back_to_local(clock, caplog):
    agent, bb, local, cloud = _cloud_agent("yes", clock)

    async def boom(system, messages):
        cloud.calls.append(messages)
        raise RuntimeError("no credits")

    cloud.reply = boom
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        assert await agent.process(message("coco who won last night?", guid="f1")) == "handled:mention"
    assert len(cloud.calls) == 1
    assert len(local.calls) == 2, "router call, then the local reply after the cloud failed"
    assert len(bb.texts) == 2 and bb.texts[1] == (ALLOWED, "yes")
    assert any("event=llm.cloud_failed" in r.getMessage() and "error=RuntimeError" in r.getMessage() for r in caplog.records)


async def test_inline_reply_to_bot_message_counts_as_mention(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    # the bot's own message comes back through the webhook with isFromMe
    assert await agent.process(message("yo", from_me=True, guid="bot-msg-1")) == "ignored:from_me"
    # a plain message replying to something else is just stored
    assert await agent.process(message("nah", guid="r0", threadOriginatorGuid="someone-elses")) == "stored"
    # an inline reply to the bot's message triggers a reply without the name
    assert await agent.process(message("why though", guid="r1", threadOriginatorGuid="bot-msg-1")) == "handled:mention"
    assert len(llm.calls) == 1 and bb.texts == [(ALLOWED, "yo")]
    # replyToGuid works too
    assert await agent.process(message("ok", guid="r2", replyToGuid="bot-msg-1")) == "handled:mention"
    assert len(llm.calls) == 2


def _reaction_agent(clock, reply, **over):
    from imessage_agent.context import ContextStore
    from tests.conftest import FakeBB, FakeLLM, make_settings

    base = dict(reaction_chance=1.0, reaction_cooldown_seconds=0)
    base.update(over)
    s = make_settings(**base)
    bb, llm = FakeBB(), FakeLLM(reply)
    ctx = ContextStore(5, 3600, clock=clock)
    agent = Agent(s, bb, llm, ctx, images=None, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    return agent, bb, llm


async def test_reaction_sent_when_model_offers_one(clock):
    agent, bb, llm = _reaction_agent(clock, "\"😂😂\"")
    assert await agent.process(message("dan fell off the boat", guid="x1")) == "stored"
    await agent.drain()
    assert bb.texts == [(ALLOWED, "😂😂")]
    assert len(llm.calls) == 1


async def test_reaction_skipped_on_none_and_long_answers(clock):
    agent, bb, llm = _reaction_agent(clock, "NONE")
    await agent.process(message("ok see you at 8", guid="x2"))
    await agent.drain()
    assert bb.texts == []
    agent, bb, llm = _reaction_agent(clock, "this is a whole paragraph of commentary that is far too long to be a reaction")
    await agent.process(message("ok see you at 8", guid="x3"))
    await agent.drain()
    assert bb.texts == []


async def test_reaction_respects_cooldown_and_default_off(clock, agent_parts):
    agent, bb, llm = _reaction_agent(clock, "lmaoo", reaction_cooldown_seconds=3600)
    await agent.process(message("one", guid="x4"))
    await agent.process(message("two", guid="x5"))
    await agent.drain()
    assert len(bb.texts) == 1
    s, bb2, llm2, ctx, agent2 = agent_parts
    assert s.reaction_chance == 0
    await agent2.process(message("funny thing happened", guid="x6"))
    await agent2.drain()
    assert bb2.texts == [] and llm2.calls == []


async def test_memory_notes_written_and_injected(clock, tmp_path):
    from imessage_agent.context import ContextStore
    from imessage_agent.memory import MemoryStore
    from tests.conftest import FakeBB, FakeLLM, make_settings

    class RecordingLLM(FakeLLM):
        def __init__(self):
            super().__init__("## People\n- Alex: lowercase, dry\n\n## Group\n- rugby jokes")
            self.systems = []

        async def reply(self, system, messages):
            self.systems.append(system)
            return await super().reply(system, messages)

    s = make_settings(memory_dir=str(tmp_path / "mem"), memory_update_every=2)
    bb, llm = FakeBB(), RecordingLLM()
    ctx = ContextStore(10, 3600, clock=clock)
    mem = MemoryStore(s.memory_dir, s.memory_max_chars, s.memory_update_every)
    agent = Agent(s, bb, llm, ctx, images=None, resolver=SenderResolver(None, {}, use_contacts=False), memory=mem)  # type: ignore[arg-type]
    await agent.process(message("hey", guid="m1"))
    await agent.drain()
    assert not mem.path_for(ALLOWED).exists(), "not yet: every 2nd message"
    await agent.process(message("rugby tonight", guid="m2"))
    await agent.drain()
    path = mem.path_for(ALLOWED)
    assert path.exists() and (path.stat().st_mode & 0o777) == 0o600
    assert (path.parent.stat().st_mode & 0o777) == 0o700
    assert "Alex: lowercase" in path.read_text()
    # the notes are part of the system prompt for the next mention, but not of the log
    await agent.process(message("coco what's up", guid="m3"))
    assert "Alex: lowercase" in llm.systems[-1] and "never read them out" in llm.systems[-1]



async def test_redelivery_with_text_after_empty_copy_is_handled(agent_parts):
    """BlueBubbles can send a message twice: first without text, then with it."""
    s, bb, llm, ctx, agent = agent_parts
    assert await agent.process(message("", guid="same")) == "ignored:empty"
    assert await agent.process(message("coco you there", guid="same")) == "handled:mention"
    assert len(llm.calls) == 1 and len(bb.texts) == 1
    # a true duplicate (text both times) is still dropped
    assert await agent.process(message("coco you there", guid="same")) == "ignored:duplicate"
    assert len(llm.calls) == 1
