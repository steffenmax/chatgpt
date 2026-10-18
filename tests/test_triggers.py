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
