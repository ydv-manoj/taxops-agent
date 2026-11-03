import json

from taxops import data, env
from taxops.metrics import score_episode
from taxops.rewards import calls_from_agenttune, episode_reward, taxops_reward
from taxops.scripted import careless, oracle, wasteful

CASES = [f"{t}-{i:05d}" for t in env.TIERS for i in range(30)]


def _reward(cid, calls):
    return episode_reward(score_episode(cid, calls), calls)


def test_reward_ordering():
    for cid in CASES:
        o, w = _reward(cid, oracle(cid)), _reward(cid, wasteful(cid))
        assert o == 2.0 and 0 < w < o, cid
    hard = [c for c in CASES if c.startswith("hard")]
    assert any(_reward(c, careless(c)) < 2.0 for c in hard)


def test_quitting_never_beats_submitting():
    cid = "medium-00003"
    assert _reward(cid, oracle(cid)[:-1]) == 0.0
    calls = wasteful(cid)
    sub = calls[-1]["arguments"]
    sub.update({k: 0 for k in sub if k not in ("case_id", "filing_status")})  # almost everything wrong
    assert _reward(cid, calls) > 0.0


def _as_agenttune(calls):
    """Shape our calls like AgentTune's rollout Trajectory (one call per step, OpenAI style)."""
    steps = []
    for c in calls:
        steps.append({"metadata": {
            "tool_calls": [{"id": "x", "type": "function",
                            "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}],
            "tool_results": [(c["name"], c["result"])]}})
    return {"steps": steps}


def test_agenttune_adapter_round_trip():
    cid = "hard-00011"
    calls = wasteful(cid)
    back = calls_from_agenttune(_as_agenttune(calls))
    assert [(c["name"], c["arguments"]) for c in back] == [(c["name"], c["arguments"]) for c in calls]
    r = taxops_reward(completions=["", ""], trajectories=[_as_agenttune(oracle(cid)), _as_agenttune(calls)],
                      case_id=[cid, cid])
    assert r[0] == 2.0 and r[1] < 2.0


def test_unknown_tool_error_dict_is_failed_call():
    traj = {"steps": [{"metadata": {"tool_calls": [{"name": "delete_everything", "arguments": {}}],
                                    "tool_results": [("delete_everything", {"error": "Tool not found"})]}}]}
    calls = calls_from_agenttune(traj)
    assert calls[0]["result"].startswith("ERROR")


def test_splits_are_disjoint_and_sized():
    sets = {k: set(v) for k, v in data.SPLITS.items()}
    assert len(sets["train"]) == 2100 and len(sets["eval"]) == 300 and len(sets["eval-shift"]) == 150
    assert not (sets["train"] & sets["eval"]) and not (sets["train"] & sets["eval-shift"])


def test_shift_cases_use_unseen_names():
    seen = set(env.EMPLOYERS + env.BANKS + env.BROKERS + env.CHARITIES + env.CLIENTS)
    for cid in data.SPLITS["eval-shift"][:30]:
        text = " ".join(d.title + d.text for d in env.generate_case(cid).documents)
        assert not any(name in text for name in seen), cid
        assert score_episode(cid, oracle(cid)).success


def test_sft_conversations_are_valid():
    rows = data.sft_conversations(data.SPLITS["train"][:60], recovery_share=0.5)
    assert {r["kind"] for r in rows} == {"oracle", "recovery"}
    for r in rows:
        msgs = r["messages"]
        assert msgs[0]["role"] == "system" and msgs[-1]["role"] == "user"
        assert all(m["role"] == "assistant" for m in msgs[2::2])
        if r["kind"] == "recovery":
            (k,) = r["skip_assistant_turns"]
            # the skipped turn is the mistake: the user turn right after it is an ERROR
            assert msgs[2 + 2 * k + 1]["content"].startswith("Tool result") and "ERROR" in msgs[2 + 2 * k + 1]["content"]
        else:
            assert r["skip_assistant_turns"] == []


def test_oracle_calculator_sums_match_the_answer():
    """The SFT demos must never teach a wrong sum."""
    from taxops.data import oracle_plan

    for cid in data.SPLITS["train"][::25] + data.SPLITS["eval-shift"][:20]:
        a = env.generate_case(cid).answer
        calcs = [c for c in oracle(cid) if c["name"] == "calculator"]
        results = {float(c["result"]) for c in calcs}
        assert not any(c["result"].startswith("ERROR") for c in calcs), cid
        assert float(a["agi"] - a["deduction"]) in results and float(a["refund_or_owed"]) in results, cid
        if " + " in " ".join(c["arguments"]["expression"] for c in calcs):
            assert float(a["agi"]) in results or a["agi"] == a["wages"], cid
        plan = oracle_plan(cid)
        assert len(plan) == 7 and plan[-1][0][0] == "submit_return", cid  # 7 turns, batched
