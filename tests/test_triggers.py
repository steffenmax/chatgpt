import pytest

from imessage_agent.triggers import detect, is_mention, parse_command


@pytest.mark.parametrize(
    "text,expected",
    [
        ("coco what's up", True),
        ("Hey COCO", True),
        ("@coco are you there?", True),
        ("coco, lunch?", True),
        ("thanks coco!", True),
        ("I love cocoa", False),  # word boundary
        ("cocos are tasty", False),
        ("decoco", False),
        ("nothing here", False),
        ("", False),
        (None, False),
    ],
)
def test_mention_detection(text, expected):
    assert is_mention(text, "Coco") is expected
    assert (detect(text, "Coco").kind == "mention") is expected


def test_multiword_bot_name():
    assert is_mention("hey Mr Bot, hi", "Mr Bot")
    assert not is_mention("mr botanist", "Mr Bot")


def test_commands():
    assert parse_command("/image a cat in space") == ("image", "a cat in space")
    assert parse_command("  /IMAGE   dog ") == ("image", "dog")
    assert parse_command("/help") == ("help", "")
    assert parse_command("/forget") == ("forget", "")
    assert parse_command("/unknown x") == (None, "")
    assert parse_command("not /image a command") == (None, "")


def test_detect_prefers_command_over_mention():
    t = detect("/image coco riding a bike", "Coco")
    assert t.kind == "image" and t.arg == "coco riding a bike"


def test_plain_message_is_none():
    assert detect("hello everyone", "Coco").kind == "none"


def test_ask_command_and_yes_no_parsing():
    from imessage_agent.llm import parse_yes_no
    from imessage_agent.triggers import parse_command

    assert parse_command("/ask who won the game") == ("ask", "who won the game")
    assert parse_command("/ASK  spaced") == ("ask", "spaced")
    assert parse_yes_no("Yes") and parse_yes_no("yes.") and parse_yes_no("<think>hm</think> yes")
    assert not parse_yes_no("no") and not parse_yes_no("yo") and not parse_yes_no("") and not parse_yes_no("yesterday")


def test_parse_intent():
    from imessage_agent.llm import parse_intent

    assert parse_intent('{"action": "chat"}').action == "chat"
    i = parse_intent('Sure! {"action": "image", "prompt": "a corgi"}')
    assert (i.action, i.prompt) == ("image", "a corgi")
    assert parse_intent('{"action": "image"}').action == "chat"  # no prompt -> chat
    i = parse_intent('{"action": "remind", "when": "at 7pm", "what": "draft"}')
    assert (i.action, i.when, i.what) == ("remind", "at 7pm", "draft")
    assert parse_intent('{"action": "remind", "when": "at 7pm"}').action == "chat"
    assert parse_intent('{"action": "nuke"}').action == "chat"
    assert parse_intent("garbage").action == "chat"


def test_image_request_gate():
    from imessage_agent.llm import looks_like_image_request as ok

    assert ok("ricky make a picture of a corgi surfing")
    assert ok("Ricky can u draw joe as a knight")
    assert ok("ricky send me a pic of doug")
    assert ok("ricky gimme a meme about mike")
    assert ok("ricky could you generate an image of the beach?")
    assert not ok("it's gonna generate another picture of doug")
    assert not ok("that pic was terrible ricky")
    assert not ok("ricky the picture looks like my uncle")
    assert not ok("ricky what did the picture look like")
    assert not ok("ricky roast joe")
    assert not ok("seriously concerned with this whole doug thing")



def test_mute_and_unmute_phrases():
    from imessage_agent.triggers import detect, parse_duration

    for t in ("coco chill", "shut up coco", "yo coco chill bro", "coco stfu lol", "coco, go away", "coco pls be quiet",
              "/mute", "coco relax for 10 min"):
        assert detect(t, "coco").kind == "mute", t
    for t in ("coco that was chill", "coco whats a chill bar near us?", "coco is enough of a clown",
              "coco chill out and tell me who won the game last night because i missed it"):
        assert detect(t, "coco").kind == "mention", t
    for t in ("/unmute", "coco come back", "coco you can talk now"):
        assert detect(t, "coco").kind == "unmute", t
    assert parse_duration("coco shut up for an hour") == 3600
    assert parse_duration("2h") == 7200 and parse_duration("1h30m") == 5400 and parse_duration("half an hour") == 1800
    assert parse_duration("10 days") == 86400 and parse_duration("coco chill") is None
