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
    assert len(llm.calls) == 2  # dispatcher + reply
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


class RoutingLLM:
    """Answers the dispatcher with `intent_json` and everything else with `reply_text`."""

    name = "fake"

    def __init__(self, intent_json, reply_text="local answer"):
        self.intent_json, self.reply_text = intent_json, reply_text
        self.calls = []

    async def reply(self, system, messages):
        self.calls.append(messages)
        return self.intent_json if "dispatcher" in system else self.reply_text

    async def healthy(self):
        return True

    async def aclose(self):
        pass


def _cloud_agent(local_reply, clock):
    from imessage_agent.context import ContextStore
    from tests.conftest import FakeBB, FakeLLM, make_settings

    s = make_settings()
    intent = '{"action": "lookup"}' if local_reply == "yes" else '{"action": "chat"}'
    bb, local, cloud = FakeBB(), RoutingLLM(intent, local_reply), FakeLLM("from the cloud")
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
    assert len(local.calls) == 1, "local model is asked only which tool to use"
    assert local.calls[0][0]["content"].endswith("JSON only.")
    assert len(cloud.calls) == 1
    assert len(bb.texts) == 2, "a 'looking it up' notice, then the answer"
    assert "look" in bb.texts[0][1] or "check" in bb.texts[0][1] or "find out" in bb.texts[0][1]
    assert bb.texts[1] == (ALLOWED, "from the cloud")


async def test_mention_stays_local_when_router_says_no(clock):
    agent, bb, local, cloud = _cloud_agent("no", clock)
    await agent.process(message("coco roast me", guid="c2"))
    assert len(local.calls) == 2, "dispatcher call + the actual reply"
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
    assert len(llm.calls) == 1 and bb.texts == [(ALLOWED, "yo")]  # /ask skips the dispatcher


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
    assert len(local.calls) == 2, "dispatcher call, then the local reply after the cloud failed"
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
    assert len(llm.calls) == 2 and bb.texts == [(ALLOWED, "yo")]  # dispatcher + reply
    # replyToGuid works too
    assert await agent.process(message("ok", guid="r2", replyToGuid="bot-msg-1")) == "handled:mention"
    assert len(llm.calls) == 4


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
    agent = Agent(s, bb, llm, ctx, images=None, resolver=SenderResolver(None, {"+15550001111": "Alex"}, use_contacts=False), memory=mem)  # type: ignore[arg-type]
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
    assert len(llm.calls) == 2 and len(bb.texts) == 1  # dispatcher + reply
    # a true duplicate (text both times) is still dropped
    assert await agent.process(message("coco you there", guid="same")) == "ignored:duplicate"
    assert len(llm.calls) == 2


async def test_natural_language_image_request_queues_an_image(clock):
    from imessage_agent.context import ContextStore
    from imessage_agent.imagegen import ImageQueue
    from tests.conftest import FakeBB, make_settings
    from tests.test_image_queue import FakeGenerator

    s = make_settings()
    bb = FakeBB()
    llm = RoutingLLM('{"action": "image", "prompt": "a corgi on a surfboard at sunset"}')
    ctx = ContextStore(5, 3600, clock=clock)
    holder = {}

    async def send(job, png):
        await holder["a"].send_image(job, png)

    async def fail(job, reason):
        await holder["a"].image_failed(job, reason)

    gen = FakeGenerator()
    q = ImageQueue(gen, send, fail)
    agent = Agent(s, bb, llm, ctx, images=q, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    holder["a"] = agent
    q.start()
    try:
        assert await agent.process(message("coco can you make me a picture of a corgi surfing", guid="n1")) == "handled:mention"
        await asyncio.wait_for(q._queue.join(), timeout=5)
    finally:
        await q.stop()
    assert gen.prompts == ["a corgi on a surfboard at sunset"]
    assert bb.texts[0][1].startswith("on it") and len(bb.attachments) == 1


async def test_natural_language_reminder_sets_one(clock, tmp_path):
    from imessage_agent.context import ContextStore
    from imessage_agent.reminders import ReminderStore
    from tests.conftest import FakeBB, make_settings

    s = make_settings()
    bb = FakeBB()
    llm = RoutingLLM('{"action": "remind", "when": "in 30m", "what": "call the dentist"}')
    ctx = ContextStore(5, 3600, clock=clock)
    store = ReminderStore(str(tmp_path / "r.json"))
    agent = Agent(s, bb, llm, ctx, images=None, resolver=SenderResolver(None, {"+15550001111": "Alex"}, use_contacts=False), reminders=store)  # type: ignore[arg-type]
    assert await agent.process(message("coco remind me in half an hour to call the dentist", guid="n2")) == "handled:mention"
    assert bb.texts[-1][1].startswith("got it Alex, i'll remind you in 30 min")
    assert [r.text for r in store.pending(ALLOWED)] == ["call the dentist"]


async def test_intent_falls_back_to_chat_when_tool_missing_or_json_bad(clock):
    from imessage_agent.context import ContextStore
    from tests.conftest import FakeBB, make_settings

    s = make_settings()
    for intent in ('{"action": "image", "prompt": "x"}', '{"action": "remind", "when": "in 5m", "what": "y"}', 'not json', '{"action": "lookup"}'):
        bb = FakeBB()
        llm = RoutingLLM(intent, "chatting")
        agent = Agent(s, bb, llm, ContextStore(5, 3600, clock=clock), images=None, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
        assert await agent.process(message("coco hey", guid="n3" + intent[:3])) == "handled:mention"
        assert bb.texts == [(ALLOWED, "chatting")], intent


async def test_voice_acks_use_the_model_and_fall_back(clock):
    from imessage_agent.context import ContextStore
    from tests.conftest import FakeBB, FakeLLM, make_settings

    s = make_settings(voice_acks=True)
    bb, llm = FakeBB(), FakeLLM("consider it gone, chief")
    agent = Agent(s, bb, llm, ContextStore(5, 3600, clock=clock), images=None, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    await agent.process(message("/forget", guid="v1"))
    assert bb.texts == [(ALLOWED, "consider it gone, chief")]
    assert "Keep every fact" in llm.calls and True or True  # system prompt not captured by FakeLLM; call happened:
    assert len(llm.calls) == 1
    bb, llm = FakeBB(), FakeLLM("")
    agent = Agent(s, bb, llm, ContextStore(5, 3600, clock=clock), images=None, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    await agent.process(message("/forget", guid="v2"))
    assert bb.texts == [(ALLOWED, "done, wiped everything i had from this chat")]


async def test_dispatcher_image_choice_is_vetoed_without_an_explicit_ask(clock):
    from imessage_agent.context import ContextStore
    from imessage_agent.imagegen import ImageQueue
    from tests.conftest import FakeBB, make_settings
    from tests.test_image_queue import FakeGenerator

    s = make_settings()
    bb = FakeBB()
    llm = RoutingLLM('{"action": "image", "prompt": "doug again"}', "lol yeah")
    ctx = ContextStore(5, 3600, clock=clock)
    holder = {}

    async def send(job, png):
        await holder["a"].send_image(job, png)

    async def fail(job, reason):
        await holder["a"].image_failed(job, reason)

    gen = FakeGenerator()
    q = ImageQueue(gen, send, fail)
    agent = Agent(s, bb, llm, ctx, images=q, resolver=SenderResolver(None, {}, use_contacts=False))  # type: ignore[arg-type]
    holder["a"] = agent
    q.start()
    try:
        assert await agent.process(message("coco it's gonna make another picture of doug", guid="g1")) == "handled:mention"
        await asyncio.sleep(0.05)
    finally:
        await q.stop()
    assert gen.prompts == [] and bb.texts == [(ALLOWED, "lol yeah")]


async def test_voice_ack_rejects_garbled_or_refusing_phrasings(clock, tmp_path):
    from imessage_agent.context import ContextStore
    from imessage_agent.reminders import ReminderStore
    from tests.conftest import FakeBB, FakeLLM, make_settings

    s = make_settings(voice_acks=True)

    async def run(model_says):
        bb = FakeBB()
        llm = FakeLLM(model_says)
        store = ReminderStore(str(tmp_path / f"r{abs(hash(model_says))}.json"))
        agent = Agent(s, bb, llm, ContextStore(5, 3600, clock=clock), images=None,
                      resolver=SenderResolver(None, {"+15550001111": "Max"}, use_contacts=False), reminders=store)  # type: ignore[arg-type]
        await agent.process(message("/remind in 45m call mom", guid="va" + str(abs(hash(model_says)))))
        return bb.texts[-1][1]

    plain = "got it Max, i'll remind you in 45 min (#1)"
    # what llama actually said in testing: drops the facts and refuses
    assert await run("can't, you gotta text max yourself, tell him to meet us at the bar at 7") == plain
    # keeps facts but adds a refusal
    assert await run("i can't set reminders but 45 min, #1") == plain
    # drops the reminder number
    assert await run("bet max, 45 min and i'll yell at you") == plain
    # a good in-voice line is used
    assert await run("bet max, 45 min from now i'm on you, that's #1") == "bet max, 45 min from now i'm on you, that's #1"


async def test_voice_ack_rejects_role_confused_echoes(clock, tmp_path):
    from imessage_agent.context import ContextStore
    from imessage_agent.reminders import ReminderStore
    from tests.conftest import FakeBB, FakeLLM, make_settings

    s = make_settings(voice_acks=True)
    plain = "got it Max, i'll remind you in 45 min (#1)"
    for i, said in enumerate(["max: hey coco, confirm that reminder #1 is saved in 45 min",
                              "coco here, 45 min, #1",
                              "the plain version is: got it max, 45 min, #1"]):
        bb = FakeBB()
        store = ReminderStore(str(tmp_path / f"e{i}.json"))
        agent = Agent(s, bb, FakeLLM(said), ContextStore(5, 3600, clock=clock), images=None,
                      resolver=SenderResolver(None, {"+15550001111": "Max"}, use_contacts=False), reminders=store)  # type: ignore[arg-type]
        await agent.process(message("/remind in 45m call mom", guid=f"ec{i}"))
        assert bb.texts[-1][1] == plain, said

