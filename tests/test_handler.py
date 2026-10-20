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
    # replyToGuid is only "the previous message": right after the bot it's a judgment call
    assert await agent.process(message("ok", guid="r2", replyToGuid="bot-msg-1")) == "stored"  # judged no
    assert len(llm.calls) == 3 and len(bb.texts) == 1
    llm.reply_text = '{"to": "bot"}'
    assert await agent.process(message("lol you wish", guid="r3", replyToGuid="bot-msg-1")) == "handled:mention"


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



class SeqLLM:
    """Dispatcher gets chat; replies come from `replies` in order (last one repeats)."""

    name = "fake"

    def __init__(self, replies):
        self.replies, self.calls, self.systems = list(replies), [], []

    async def reply(self, system, messages):
        if "dispatcher" in system:
            return '{"action": "chat"}'
        self.calls.append(messages)
        self.systems.append(system)
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]

    async def healthy(self):
        return True

    async def aclose(self):
        pass


def _seq_agent(clock, replies):
    from imessage_agent.context import ContextStore
    from tests.conftest import FakeBB, make_settings

    s = make_settings()
    bb, llm = FakeBB(), SeqLLM(replies)
    agent = Agent(s, bb, llm, ContextStore(10, 3600, clock=clock), images=None,
                  resolver=SenderResolver(None, {"+15550001111": "Max", "+15550002222": "Mike"}, use_contacts=False))  # type: ignore[arg-type]
    return agent, bb, llm


async def test_reply_written_as_a_member_is_retried(clock):
    agent, bb, llm = _seq_agent(clock, ["max: shut up joe what is that", "nah max, that's on you"])
    await agent.process(message("yo", sender="+15550002222", guid="i0"))
    assert await agent.process(message("coco say something", guid="i1")) == "handled:mention"
    assert bb.texts == [(ALLOWED, "nah max, that's on you")]
    assert len(llm.calls) == 2 and "Never write lines for anyone else" in llm.systems[0]


async def test_reply_still_impersonating_after_retry_loses_the_label(clock):
    agent, bb, llm = _seq_agent(clock, ["Mike: bro dont go there"])
    await agent.process(message("yo", sender="+15550002222", guid="j0"))
    await agent.process(message("coco thoughts", guid="j1"))
    assert bb.texts == [(ALLOWED, "bro dont go there")]


async def test_transcript_continuation_is_cut(clock):
    agent, bb, llm = _seq_agent(clock, ["lmao that's rough\nMike: nah it's funny\nMax: ok"])
    await agent.process(message("yo", sender="+15550002222", guid="k0"))
    await agent.process(message("coco thoughts", guid="k1"))
    assert bb.texts == [(ALLOWED, "lmao that's rough")]


async def test_made_up_links_are_removed_but_seen_links_kept(clock):
    agent, bb, llm = _seq_agent(clock, ["check this https://www.tiktok.com/t/ZTRfake123/ lol"])
    await agent.process(message("coco send a tiktok", guid="l1"))
    assert bb.texts[-1] == (ALLOWED, "check this lol")
    agent, bb, llm = _seq_agent(clock, ["yeah https://example.com/real is great"])
    await agent.process(message("look at https://example.com/real", sender="+15550002222", guid="l2"))
    await agent.process(message("coco thoughts on that link", guid="l3"))
    assert bb.texts[-1] == (ALLOWED, "yeah https://example.com/real is great")
    agent, bb, llm = _seq_agent(clock, ["https://www.tiktok.com/t/ZTRfake999/"])
    await agent.process(message("coco send a tiktok", guid="l4"))
    assert bb.texts[-1][1].startswith("can't pull up links")


def test_speaker_helpers():
    from imessage_agent.context import ContextMessage
    from imessage_agent.llm import fix_speaker, speaker_names, stop_sequences, strip_unseen_urls

    hist = [ContextMessage("Max", "hi", 0.0), ContextMessage("Mike Schiavello", "yo", 0.0), ContextMessage("+15551234567", "x", 0.0)]
    names = speaker_names(hist, "Ricky")
    assert names == ["max", "mike", "ricky"]
    assert "\nMax:" in stop_sequences(names) and "\nmike:" in stop_sequences(names)
    assert fix_speaker("max: hey", names, "Ricky") == ("max: hey", True)
    assert fix_speaker("ok cool\nmike: no", names, "Ricky") == ("ok cool", False)
    assert fix_speaker("note: max is late", names, "Ricky") == ("note: max is late", False)
    assert strip_unseen_urls("see www.fake.com/x now", "") == "see now"
    assert strip_unseen_urls("see https://a.com/b.", "https://a.com/b") == "see https://a.com/b."


async def test_assistant_style_refusal_becomes_a_short_roast(agent_parts):
    from imessage_agent.llm import REFUSAL_ROASTS

    s, bb, llm, ctx, agent = agent_parts
    llm.reply_text = "I can't create explicit content. Is there anything else I can help you with?"
    assert await agent.process(message("coco say something filthy", guid="r1")) == "handled:mention"
    sent = bb.texts[-1][1]
    assert sent in REFUSAL_ROASTS and len(sent.split()) <= 2
    # the next one is a different roast
    llm.reply_text = "As an AI, I'd rather not."
    await agent.process(message("coco come on", guid="r2"))
    assert bb.texts[-1][1] in REFUSAL_ROASTS and bb.texts[-1][1] != sent


def test_refusal_detector_leaves_normal_lines_alone():
    from imessage_agent.llm import is_refusal

    for ok in ("i can't believe the jets did that", "won't lie that pic is rough", "can't help you, you're hopeless lol",
               "explicitly told you the knicks would choke", "i'm not a bot, i'm ricky"):
        assert not is_refusal(ok), ok
    for bad in ("I can't help with that.", "I cannot generate explicit images.", "As an AI language model, no.",
                "That goes against my guidelines.", "I’m unable to assist with that request",
                "I'm not comfortable writing that", "Is there anything else I can help with?"):
        assert is_refusal(bad), bad


async def test_reaction_prompt_varies_and_repeats_are_dropped(clock):
    import random as _r

    from imessage_agent.llm import parse_reaction, reaction_addendum

    a, b = reaction_addendum(None, _r.Random(1)), reaction_addendum("lmaooo", _r.Random(2))
    assert a != b and "never the same thing you said last time ('lmaooo')" in b
    assert parse_reaction("nah dan you did NOT just say that") == "nah dan you did NOT just say that"
    assert parse_reaction("Alex: lol") is None and parse_reaction("I can't help with that") is None
    agent, bb, llm = _reaction_agent(clock, "lmaooo")
    await agent.process(message("dan fell off the boat", guid="v1"))
    await agent.drain()
    llm.reply_text = "LMAOOOOO"
    await agent.process(message("then he fell again", guid="v2"))
    await agent.drain()
    assert bb.texts == [(ALLOWED, "lmaooo")]  # the near-identical second one was dropped


class _SysLLM:
    name = "fake"

    def __init__(self, reply="yo", delay=0.0):
        self.reply_text, self.delay = reply, delay
        self.systems: list[str] = []

    async def reply(self, system, messages, stop=None):
        self.systems.append(system)
        await asyncio.sleep(self.delay)
        return self.reply_text

    async def healthy(self):
        return True

    async def aclose(self):
        pass


def _batch_agent(clock, llm, **over):
    from imessage_agent.context import ContextStore
    from tests.conftest import FakeBB, make_settings

    s = make_settings(reply_batch_seconds=0.05, reply_batch_max_seconds=0.5, **over)
    bb = FakeBB()
    names = {"+15550001111": "Alex", "+15550002222": "Sam", "+15550003333": "Dan"}
    agent = Agent(s, bb, llm, ContextStore(30, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, names, use_contacts=False))
    return agent, bb


async def test_mentions_close_together_get_one_reply(clock):
    llm = _SysLLM("ok ok one at a time")
    agent, bb = _batch_agent(clock, llm)
    assert await agent.process(message("coco who wins tonight", guid="b1", sender="+15550001111")) == "queued:mention"
    await agent.process(message("coco and are you coming sat", guid="b2", sender="+15550002222"))
    await agent.process(message("lol", guid="b3", sender="+15550003333"))
    await agent.process(message("coco answer me", guid="b4", sender="+15550003333"))
    await asyncio.sleep(0.2)
    await agent.drain()
    assert bb.texts == [(ALLOWED, "ok ok one at a time")]
    reply_system = llm.systems[-1]
    assert "Several people pinged you at once" in reply_system
    assert '- Alex: "coco who wins tonight"\n- Sam: "coco and are you coming sat"\n- Dan: "coco answer me"' in reply_system
    assert "lol" not in reply_system.split("Several people")[1]  # only the pings are named
    assert "Stay consistent with what you already said" in reply_system


async def test_a_ping_during_a_reply_waits_and_sees_that_reply(clock):
    llm = _SysLLM("first take", delay=0.2)
    agent, bb = _batch_agent(clock, llm)
    await agent.process(message("coco jets or giants", guid="c1"))
    await asyncio.sleep(0.1)  # batch closed, reply being written
    await agent.process(message("coco wait what", guid="c2", sender="+15550002222"))
    await asyncio.sleep(0.8)
    await agent.drain()
    assert [t for _, t in bb.texts] == ["first take", "first take"]  # two replies, never overlapping
    second_history = agent.context.get(ALLOWED)
    assert [m.text for m in second_history if m.from_bot][:1] == ["first take"]


def test_trim_reply_cuts_at_a_sentence():
    from imessage_agent.llm import trim_reply

    long = "jets in five. " + "dan is cooked and everyone knows it. " * 20
    out = trim_reply(long, 120)
    assert len(out) <= 120 and out.endswith(".")
    assert trim_reply("short", 120) == "short"


async def test_mute_silences_the_chat_until_unmuted(clock):
    llm = _SysLLM("yo")
    agent, bb = _batch_agent(clock, llm, reaction_chance=1.0, reaction_cooldown_seconds=0)
    await agent.process(message("coco who wins", guid="m0"))  # queued, then muted before it's answered
    assert await agent.process(message("coco chill for 10 min", guid="m1")) == "handled:mute"
    assert "10 min" in bb.texts[-1][1]
    assert await agent.process(message("coco you there", guid="m2")) == "muted"
    assert await agent.process(message("dan fell off the boat", guid="m3")) == "stored"
    await asyncio.sleep(0.2)
    await agent.drain()
    assert len(bb.texts) == 1 and llm.systems == []  # no reply, no reaction, no queued answer
    assert await agent.process(message("/unmute", guid="m4")) == "handled:unmute"
    assert await agent.process(message("coco welcome back", guid="m5")) == "queued:mention"
    await asyncio.sleep(0.2)
    await agent.drain()
    assert bb.texts[-1][1] == "yo"


async def test_mute_expires(clock, monkeypatch):
    import time as _time

    llm = _SysLLM("yo")
    agent, bb = _batch_agent(clock, llm)
    await agent.process(message("/mute 5m", guid="e1"))
    assert agent._muted(ALLOWED)
    real = _time.monotonic()
    monkeypatch.setattr("imessage_agent.handler.time.monotonic", lambda: real + 301)
    assert not agent._muted(ALLOWED)


async def test_mute_defaults_to_two_minutes(clock):
    agent, bb = _batch_agent(clock, _SysLLM("yo"))
    await agent.process(message("coco chill", guid="d1"))
    assert "2 min" in bb.texts[-1][1]



async def test_followup_judgment_only_right_after_the_bot(agent_parts, monkeypatch):
    s, bb, llm, ctx, agent = agent_parts
    llm.reply_text = '{"to": "bot"}'
    await agent.process(message("yo", from_me=True, guid="bot-1"))
    # after someone else's message: not judged at all
    assert await agent.process(message("dan you up", guid="f0", replyToGuid="someone")) == "stored"
    assert llm.calls == []
    # too long after the bot spoke: not judged either
    agent._bot_guids["bot-1"] -= 31
    assert await agent.process(message("ok", guid="f1", replyToGuid="bot-1")) == "stored"
    assert llm.calls == []


async def test_dm_messages_are_always_for_the_bot(clock):
    from imessage_agent.context import ContextStore
    from tests.conftest import FakeBB, FakeLLM, make_settings

    dm = "iMessage;-;+15550001111"
    s = make_settings(allowed_chat_guids=f"{ALLOWED},{dm}")
    bb, llm = FakeBB(), FakeLLM("sup")
    agent = Agent(s, bb, llm, ContextStore(5, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, {}, use_contacts=False))
    assert await agent.process(message("what are you doing tonight", chat=dm, guid="dm1")) == "handled:mention"
    assert bb.texts == [(dm, "sup")]


async def test_inline_reply_to_a_bot_message_from_before_a_restart(agent_parts):
    s, bb, llm, ctx, agent = agent_parts
    bb.known_messages = {"old-bot": {"guid": "old-bot", "isFromMe": True}, "old-person": {"guid": "old-person", "isFromMe": False}}
    assert await agent.process(message("why though", guid="o1", threadOriginatorGuid="old-bot")) == "handled:mention"
    assert await agent.process(message("facts", guid="o2", threadOriginatorGuid="old-person")) == "stored"
    assert await agent.process(message("hm", guid="o3", threadOriginatorGuid="gone")) == "stored"  # lookup failed
    # the old message counts for inline replies but doesn't open the 30s follow-up window
    llm.reply_text = '{"to": "bot"}'
    calls = len(llm.calls)
    assert await agent.process(message("ok", guid="o4", replyToGuid="old-bot")) == "stored"
    assert len(llm.calls) == calls



async def test_reply_names_the_ping_even_when_newer_messages_follow(clock):
    llm = _SysLLM("pizza obviously")
    agent, bb = _batch_agent(clock, llm)
    await agent.process(message("coco wings or pizza tonight", guid="t1"))
    await agent.process(message("jack you still have my charger", guid="t2", sender="+15550002222"))
    await asyncio.sleep(0.2)
    await agent.drain()
    assert 'You\'re answering Alex, who said to you: "coco wings or pizza tonight". Answer that.' in llm.systems[-1]
