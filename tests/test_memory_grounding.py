from imessage_agent.context import ContextMessage
from imessage_agent.llm import memory_messages
from imessage_agent.memory import MemoryStore, ground_notes


NOTES = """## People
* Max: lowercase, dry, organizes the plans
- Mike Schiavello: loud, roasts everyone
* Doug: a legend in his own mind, superglued shoes to the floor
- Joe: quiet, shows up with one-liners

## Group
- rugby jokes every week
"""


def test_people_section_only_keeps_real_senders():
    out = ground_notes(NOTES, ["Max", "Mike Schiavello", "Joe"])
    people = out.split("## Running bits")[0]
    assert "Max:" in people and "Mike Schiavello:" in people and "Joe:" in people
    assert "Doug" not in people
    assert "- Doug (not a member, a bit): a legend in his own mind, superglued shoes to the floor" in out
    assert out.index("## People") < out.index("## Running bits") < out.index("## Group")
    assert "rugby jokes every week" in out


def test_first_name_matching_and_missing_sections():
    out = ground_notes("## People\n- mike: yells\n- Sam: ?\n", ["Mike Schiavello", "Max"])
    assert "- mike: yells" in out and "Sam (not a member" in out and "## Group" in out
    assert ground_notes("no sections here", ["Max"]) == ""
    out = ground_notes("## People\n- Max: ok\n## Running bits\n- Doug (not a member, a bit): old\n", ["Max"])
    assert out.count("Doug") == 1


def test_bot_lines_are_marked_and_members_listed():
    hist = [ContextMessage("Max", "hey", 0.0), ContextMessage("Ricky", "my buddy doug says hi", 0.0, from_bot=True), ContextMessage("Joe", "who", 0.0)]
    content = memory_messages("", hist, "Ricky", ["Max", "Joe"])[0]["content"]
    assert "doug" not in content and "Max: hey\nJoe: who" in content  # the bot's own lines never reach the note-taker
    assert "the only names allowed under ## People): Max, Joe" in content


async def test_store_update_grounds_model_output(tmp_path):
    class LLM:
        name = "fake"

        async def reply(self, system, messages):
            assert "own messages are left out" in system and "never go in" in system and "Running bits" in system
            return "## People\n- Max: dry\n- Doug: invented\n## Group\n- bits\n"

    st = MemoryStore(str(tmp_path / "m"), 3000, 2)
    hist = [ContextMessage("Max", "hi", 0.0), ContextMessage("Ricky", "doug!", 0.0, from_bot=True)]
    assert await st.update("c", "Ricky", hist, LLM())
    notes = st.load("c")
    assert "- Max: dry" in notes.split("## Running bits")[0]
    assert "Doug (not a member, a bit): invented" in notes


async def test_pinned_section_survives_rewrites_and_quiet_members_stay(tmp_path):
    from imessage_agent.context import ContextMessage
    from imessage_agent.memory import MemoryStore

    class Writer:
        name = "fake"

        def __init__(self):
            self.seen = ""

        async def reply(self, system, messages):
            self.seen = messages[0]["content"]
            return ("## Pinned\n* Dziuba: actually a genius\n\n## People\n* Max: runs the chat\n"
                    "* Dziuba: quiet today\n* Doug: made up\n\n## Running bits\n\n## Group\n* lots of NY sports")

    store = MemoryStore(str(tmp_path), 3000, 1)
    store.save("g", "## Pinned\n* Dziuba: call him Dziuba; the butt of every joke\n\n## People\n* Max: runs the chat\n")
    w = Writer()
    history = [ContextMessage("Max", "yo", 0.0)]
    assert await store.update("g", "Ricky", history, w, participants=["Max", "Dziuba", "Tom Smith"])
    notes = store.load("g")
    assert notes.startswith("## Pinned\n* Dziuba: call him Dziuba; the butt of every joke\n\n## People")
    assert "actually a genius" not in notes and "butt of every joke" not in w.seen  # the model never rewrites it
    assert "* Dziuba: quiet today" in notes  # in the chat, so still a member
    assert "Doug (not a member, a bit)" in notes
    assert "Dziuba, Tom Smith" in w.seen
