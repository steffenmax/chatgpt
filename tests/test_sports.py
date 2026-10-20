import asyncio
from datetime import datetime, timedelta, timezone

from imessage_agent.alerts import SportsAlerts, parse_teams
from imessage_agent.bookie import Bookie, parse_bet
from imessage_agent.context import ContextStore
from imessage_agent.handler import Agent, SenderResolver
from imessage_agent.sports import Game, ScoringPlay, Team, TeamIndex, american_profit, parse_scoreboard
from tests.conftest import ALLOWED, FakeBB, FakeLLM, make_settings, message

NOW = datetime.now(timezone.utc)


def game(gid="g1", home=("HOU", "Texans", "Houston Texans", -535), away=("NYG", "Giants", "New York Giants", 400),
         state="pre", hs=0, as_=0, start=None, detail="", period=0, completed=False):
    return Game(gid, start or NOW + timedelta(days=2), state, detail, period, "", Team(home[0], home[1], home[2], hs, home[3]),
                Team(away[0], away[1], away[2], as_, away[3]), week=7, completed=completed)


def board(games):
    return {"week": {"number": 7}, "season": {"type": 2}, "events": [{
        "id": g.id, "date": g.start.isoformat().replace("+00:00", "Z"),
        "status": {"period": g.period, "displayClock": "4:12", "type": {"state": g.state, "shortDetail": g.detail, "completed": g.completed}},
        "competitions": [{"competitors": [
            {"homeAway": "home", "score": str(g.home.score), "team": {"abbreviation": g.home.abbr, "name": g.home.name, "displayName": g.home.full}},
            {"homeAway": "away", "score": str(g.away.score), "team": {"abbreviation": g.away.abbr, "name": g.away.name, "displayName": g.away.full}}],
            "odds": [{"details": "HOU -10", "overUnder": 42.5, "moneyline": {"home": {"close": {"odds": str(g.home.moneyline)}},
                                                                             "away": {"close": {"odds": f"+{g.away.moneyline}"}}}}]}],
    } for g in games]}


class FakeESPN:
    def __init__(self, games, plays=None):
        self.games, self.plays = games, plays or {}
        self.fetched = []

    async def scoreboard(self, week=None, season_type=None, league="nfl", date=None):
        self.fetched.append((league, week, date))
        return [g for g in self.games if g.league == league]

    async def upcoming(self, league="nfl"):
        return [g for g in self.games if g.league == league]

    async def upcoming_all(self, leagues=None):
        return list(self.games)

    async def scoring_plays(self, gid):
        return list(self.plays.get(gid, []))

    async def aclose(self):
        pass


# -- parsing and math --
def test_scoreboard_parsing_odds_and_payouts():
    g = parse_scoreboard(board([game()]))[0]
    assert g.matchup == "Giants @ Texans" and g.home.moneyline == -535 and g.away.moneyline == 400 and g.spread == "HOU -10"
    assert american_profit(100, 400) == 400 and american_profit(535, -535) == 100 and american_profit(50, None) == 50
    idx = TeamIndex.from_games([game(), game("g2", ("PHI", "Eagles", "Philadelphia Eagles", -150), ("NYJ", "Jets", "New York Jets", 130))])
    assert idx.find("big blue") == ("nfl", "NYG") and idx.find("the jets") == ("nfl", "NYJ") and idx.find("birds") == ("nfl", "PHI")
    assert idx.find("new york") is None
    assert parse_teams("NYG=Max, Joe;NYJ=Chris Oddo") == {"NYG": ["Max", "Joe"], "NYJ": ["Chris Oddo"]}
    assert parse_bet("ricky 50 on the jets") == (50, "the jets") and parse_bet("ricky all in on pats!") == ("all", "pats")
    assert parse_bet("ricky bet 20 bucks on big blue") == (20, "big blue") and parse_bet("ricky who's on first") is None
    assert parse_bet("50 jets") is None and parse_bet("50 jets", loose=True) == (50, "jets")
    assert parse_bet("ricky 50 cent is a legend") is None


# -- alerts --
async def test_alerts_never_backfill_then_announce_new_plays_half_and_final(tmp_path):
    td = ScoringPlay("p1", "NYG", "TD", "Nabers 40 yd pass from Dart", 2, "4:12", 0, 7)
    espn = FakeESPN([game(state="in", period=2, as_=7)], {"g1": [td]})
    sent = []

    async def on_alert(a):
        sent.append(a)

    path = tmp_path / "state.json"
    sa = SportsAlerts(espn, {"NYG": ["Max", "Joe"]}, str(path), on_alert=on_alert)
    assert await sa.tick() == []  # first sight of a game in progress: no backfill
    fg = ScoringPlay("p2", "HOU", "FG", "Fairbairn 48 yd field goal", 2, "0:03", 3, 7)
    espn.plays["g1"].append(fg)
    await sa.tick()
    assert [a.kind for a in sent] == ["score"] and sent[0].text.startswith("🏈 FG Texans: Fairbairn 48 yd field goal. Giants 7, Texans 3 (Q2 0:03)")
    assert sent[0].fans == {"NYG": ["Max", "Joe"]}
    espn.games = [game(state="in", period=2, as_=7, hs=3, detail="Halftime")]
    await sa.tick()
    assert sent[-1].kind == "half" and sent[-1].text == "Halftime: Giants 7, Texans 3."
    # a restart reads the state file: nothing repeats
    sa2 = SportsAlerts(espn, {"NYG": ["Max"]}, str(path), on_alert=on_alert)
    await sa2.tick()
    assert len(sent) == 2
    espn.games = [game(state="post", period=4, as_=24, hs=10, completed=True)]
    await sa2.tick()
    assert sent[-1].kind == "final" and sent[-1].text == "FINAL: Giants 24, Texans 10." and sent[-1].scorer == "NYG"
    assert path.stat().st_mode & 0o777 == 0o600


async def test_kickoff_and_unfollowed_games(tmp_path):
    espn = FakeESPN([game(), game("g9", ("DAL", "Cowboys", "Dallas Cowboys", -110), ("KC", "Chiefs", "Kansas City Chiefs", -110))])
    sent = []

    async def on_alert(a):
        sent.append(a)

    sa = SportsAlerts(espn, {"NYG": ["Max"]}, str(tmp_path / "s.json"), on_alert=on_alert)
    await sa.tick()
    espn.games = [game(state="in", period=1), game("g9", ("DAL", "Cowboys", "Dallas Cowboys", -110), ("KC", "Chiefs", "Kansas City Chiefs", -110), state="in")]
    await sa.tick()
    assert [a.text for a in sent] == ["🏈 Giants @ Texans just kicked off."]
    assert sa.next_wait(espn.games) == 45.0


# -- bookie --
def test_book_places_settles_and_bails_out(tmp_path):
    b = Bookie(str(tmp_path / "book.json"), start=1000, bailout=100, clock=lambda: 1000.0)
    g = game()
    bet = b.place(ALLOWED, "+1", "Max", g, "NYG", 100)
    assert bet["to_win"] == 400 and bet["odds"] == 400 and b.balance(ALLOWED, "+1", "Max") == 900
    assert b.place(ALLOWED, "+2", "Joe", g, "HOU", 2000) == "too_much"
    all_in = b.place(ALLOWED, "+2", "Joe", g, "HOU", "all")
    assert all_in["stake"] == 1000 and all_in["to_win"] == 186
    assert b.place(ALLOWED, "+3", "Tom", game(state="in"), "NYG", 10) == "started"
    assert b.open_game_ids() == {"nfl:g1"}
    res = b.settle([game(state="post", as_=24, hs=10, completed=True)])[ALLOWED]
    assert [(r.name, r.result, r.delta, r.balance) for r in res] == [("Max", "won", 400, 1400), ("Joe", "lost", -1000, 0)]
    assert b.settle([game(state="post", as_=24, hs=10, completed=True)]) == {}  # once
    assert b.bailout(ALLOWED, "+2", "Joe") == (True, 100) and b.bailout(ALLOWED, "+2", "Joe")[0] is False
    assert b.bailout(ALLOWED, "+1", "Max")[0] is False  # not broke
    board_rows = b.leaderboard(ALLOWED)
    assert [r["name"] for r in board_rows] == ["Max", "Tom", "Joe"] and board_rows[-1]["bailouts"] == 1
    tie = b.place(ALLOWED, "+1", "Max", game("g2"), "NYG", 50)
    b.settle([game("g2", state="post", as_=10, hs=10, completed=True)])
    assert b.balance(ALLOWED, "+1", "Max") == 1400 and tie["status"] == "push"
    assert (tmp_path / "book.json").stat().st_mode & 0o777 == 0o600


# -- in the chat --
def _agent(clock, games, tmp_path, llm_reply="giants fans eating good tonight"):
    s = make_settings(sports_chat_guids=ALLOWED, bookie_enabled=True)
    bb, llm = FakeBB(), FakeLLM(llm_reply)
    agent = Agent(s, bb, llm, ContextStore(10, 3600, clock=clock), images=None,  # type: ignore[arg-type]
                  resolver=SenderResolver(None, {"+15550001111": "Max"}, use_contacts=False),
                  espn=FakeESPN(games), bookie=Bookie(str(tmp_path / "book.json")), sports_fans={"NYG": ["Max"], "HOU": ["Joe"]})
    return agent, bb, llm


async def test_betting_in_the_chat(clock, tmp_path):
    agent, bb, llm = _agent(clock, [game()], tmp_path)
    assert await agent.process(message("coco 100 on big blue", guid="b1")) == "handled:bet"
    assert bb.texts[-1][1].startswith("locked: Max 100 on the Giants (+400) vs the Texans ") and "pays 400. 900 left." in bb.texts[-1][1]
    await agent.process(message("/bet 50 texans", guid="b2"))
    assert "betting against your own team. cold blooded" in bb.texts[-1][1]
    await agent.process(message("/bets", guid="b3"))
    assert bb.texts[-1][1] == "riding:\nMax 100 on Giants (+400) to win 400\nMax 50 on Texans (-535) to win 9"
    await agent.process(message("/odds", guid="b4"))
    assert bb.texts[-1][1].startswith("NFL lines (moneyline):\nGiants +400 @ Texans -535")
    await agent.process(message("/bet 50 dolphins", guid="b5"))
    assert bb.texts[-1][1] == "who? i don't have them on the board"
    await agent.on_games([game(state="post", as_=24, hs=10, completed=True)])
    msg = bb.texts[-1][1]
    assert msg.startswith("💰 book settled:\n✅ Max +400 on the Giants → 1350\n❌ Max -50 on the Texans → 1350")
    assert msg.endswith("giants fans eating good tonight")


async def test_alert_comment_is_guarded(clock, tmp_path):
    from imessage_agent.alerts import Alert

    g = game(state="in", as_=7)
    agent, bb, llm = _agent(clock, [g], tmp_path, llm_reply="giants by 40 easy")  # invents a number: dropped
    await agent.on_alert(Alert("score", "🏈 TD Giants: pass. Giants 7, Texans 0 (Q1 9:00)", g, scorer="NYG", fans={"NYG": ["Max"]}))
    assert bb.texts == [(ALLOWED, "🏈 TD Giants: pass. Giants 7, Texans 0 (Q1 9:00)")]
    llm.reply_text = "MAX WE'RE SO BACK"
    await agent.on_alert(Alert("score", "🏈 TD Giants: run. Giants 14, Texans 0 (Q2 9:00)", g, scorer="NYG", fans={"NYG": ["Max"]}))
    assert bb.texts[-1][1].endswith("\nMAX WE'RE SO BACK")
    await agent.process(message("/mute 10m", guid="m"))
    n = len(bb.texts)
    await agent.on_alert(Alert("final", "FINAL: Giants 14, Texans 0.", g, scorer="NYG"))
    assert len(bb.texts) == n  # muted
    _ = asyncio  # keep import used


def test_fan_lookup_handles_two_chrises(clock, tmp_path):
    agent, bb, llm = _agent(clock, [game()], tmp_path)
    agent.sports_fans = {"NYG": ["Max Steffen", "Chris Schiavello"], "NYJ": ["Chris Oddo"], "NE": ["Dziuba"]}
    assert agent._fan_of("Chris Oddo") == "NYJ" and agent._fan_of("Chris Schiavello") == "NYG"
    assert agent._fan_of("Chris") is None and agent._fan_of("Max") == "NYG" and agent._fan_of("Dziuba") == "NE"



def nba(gid="n1", state="pre", hs=0, as_=0, completed=False):
    g = game(gid, ("NY", "Knicks", "New York Knicks", -180), ("PHI", "76ers", "Philadelphia 76ers", 150), state=state,
             hs=hs, as_=as_, completed=completed, start=NOW + timedelta(hours=3))
    g.league = "nba"
    return g


def nhl_jets():
    g = game("h1", ("STL", "Blues", "St. Louis Blues", -135), ("WPG", "Jets", "Winnipeg Jets", 114))
    g.league = "nhl"
    return g


async def test_bets_across_leagues(clock, tmp_path):
    nyj = game("g5", ("NYJ", "Jets", "New York Jets", -238), ("MIA", "Dolphins", "Miami Dolphins", 195))
    sfg = game("m1", ("SF", "Giants", "San Francisco Giants", 120), ("LAD", "Dodgers", "Los Angeles Dodgers", -140))
    sfg.league = "mlb"
    agent, bb, llm = _agent(clock, [game(), nyj, sfg, nba(), nhl_jets()], tmp_path)
    await agent.process(message("coco 100 on the knicks", guid="k1"))
    assert bb.texts[-1][1].startswith("locked: Max 100 on the Knicks (-180) vs the 76ers ")
    await agent.process(message("/bet 50 jets", guid="k2"))  # NFL and NHL Jets, neither followed here
    assert bb.texts[-1][1].startswith("which one, NFL or NHL?")
    await agent.process(message("/bet 50 nhl jets", guid="k3"))
    assert bb.texts[-1][1].startswith("locked: Max 50 on the Jets (+114) vs the Blues ")
    await agent.process(message("/bet 20 giants", guid="k4"))  # NFL Giants are followed: no question
    assert bb.texts[-1][1].startswith("locked: Max 20 on the Giants (+400) vs the Texans ")
    await agent.process(message("/odds nba", guid="k5"))
    assert bb.texts[-1][1].startswith("NBA lines (moneyline):\n76ers +150 @ Knicks -180") and "(NBA, " in bb.texts[-1][1]
    # the Knicks game ends: settled from that day's NBA scoreboard
    agent.espn.games = [nba(state="post", hs=112, as_=99, completed=True)]
    await agent.on_games([])
    assert any(f[0] == "nba" and f[2] for f in agent.espn.fetched)
    assert "✅ Max +55 on the Knicks" in bb.texts[-1][1]
