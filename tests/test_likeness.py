import json

from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver, _norm_addr
from imessage_agent.likeness import Likenesses, blocked
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message


def _setup(tmp_path, opted_in=True):
    d = tmp_path / "likeness" / "dan"
    d.mkdir(parents=True)
    (d / "dan.safetensors").write_bytes(b"lora")
    (d / "likeness.json").write_text(json.dumps({"name": "dziuba", "aliases": ["dziuba", "matt dziuba"],
                                                  "lora": "dan.safetensors", "scale": 1.0, "owner": "+15550003333",
                                                  "opted_in": opted_in}))
    return Likenesses(str(tmp_path / "likeness"), norm=_norm_addr)


class Q:
    def __init__(self):
        self.jobs = []

    def qsize(self):
        return 0

    async def submit(self, job):
        self.jobs.append(job)


def test_matching_and_blocking(tmp_path):
    lk = _setup(tmp_path)
    assert lk.match("dziuba as a hot dog").name == "dziuba" and lk.match("a hot dog") is None
    assert lk.match("Matt Dziuba!! on the moon") is not None and lk.match("dziubaesque") is None
    for bad in ("dziuba naked on a beach", "dziuba in lingerie", "dziuba dead in a ditch", "sexy dziuba", "dziuba mugshot"):
        assert blocked(bad), bad
    for ok in ("dziuba as a hot dog", "dziuba riding a horse into battle", "dziuba crying at a jets game"):
        assert not blocked(ok), ok
    assert _setup(tmp_path / "x", opted_in=False).match("dziuba as a hot dog") is None


async def test_image_request_uses_the_likeness_and_respects_opt_out(clock, tmp_path):
    lk = _setup(tmp_path)
    bb, q = FakeBB(), Q()
    agent = Agent(make_settings(), bb, FakeLLM(), ContextStore(10, 3600, clock=clock), images=q,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, {}, use_contacts=False), likenesses=lk)
    await agent.process(message("/image dziuba as a hot dog at a barbecue", guid="i1"))
    job = q.jobs[-1]
    assert job.prompt == "dziuba, a man. dziuba as a hot dog at a barbecue"
    assert job.loras == ((str(tmp_path / "likeness" / "dan" / "dan.safetensors"), 1.0),)
    await agent.process(message("/image dziuba naked", guid="i2"))
    from imessage_agent.handler import LIKENESS_NO

    assert len(q.jobs) == 1 and bb.texts[-1][1] in [n.replace("{name}", "dziuba") for n in LIKENESS_NO]
    # only the person can opt out
    await agent.process(message("/optout", guid="o1", sender="+15550001111"))
    assert bb.texts[-1][1] == "you're not in my picture book, nothing to opt out of" and lk.people[0].opted_in
    await agent.process(message("/optout", guid="o2", sender="+15550003333"))
    assert lk.people[0].opted_in is False
    assert json.loads((tmp_path / "likeness" / "dan" / "likeness.json").read_text())["opted_in"] is False
    await agent.process(message("/image dziuba as a hot dog", guid="i3"))
    assert q.jobs[-1].loras == () and q.jobs[-1].prompt == "dziuba as a hot dog"  # just a random guy now
    await agent.process(message("/optin", guid="o3", sender="+15550003333"))
    assert lk.people[0].opted_in is True


def test_mflux_passes_the_lora_unbaked():
    from pathlib import Path

    from imessage_agent.imagegen import MfluxGenerator

    argv = MfluxGenerator("mflux-generate-z-image-turbo", "m")._argv(None, "p", Path("o.png"), 1, (("/x/dan.safetensors", 1.0),))
    assert argv[-4:] == ["--lora", "/x/dan.safetensors", "1.0", "--no-bake-lora"]
