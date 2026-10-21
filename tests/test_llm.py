

def test_jabs_at_bot():
    from imessage_agent.llm import jabs_at_bot
    for t in ["oh shut up ricky", "ricky stfu", "ur so trash ricky", "ricky you're an idiot", "nobody asked ricky",
              "ricky you suck", "L ricky", "ricky is so cringe", "fuck you ricky"]:
        assert jabs_at_bot(t), t
    for t in ["ricky what time is the giants game", "ricky who's starting at qb", "this game is wild ricky",
              "ricky book carbone saturday"]:
        assert not jabs_at_bot(t), t
