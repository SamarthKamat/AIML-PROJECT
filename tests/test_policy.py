from ml.decision.policy import ACTIONS, break_even_probabilities, choose_action


def test_zero_exposure_means_routine_monitoring():
    assert choose_action(0.9, 0.0)["action"] == "routine_monitoring"


def test_action_escalates_with_probability():
    order = list(ACTIONS)
    prev = 0
    for p in [0.0, 0.05, 0.2, 0.5, 0.9]:
        k = order.index(choose_action(p, 2_000_000)["action"])
        assert k >= prev
        prev = k
    assert prev == len(order) - 1


def test_break_even_matches_decision():
    be = break_even_probabilities(1_000_000)["routine_monitoring->enhanced_monitoring"]
    assert choose_action(be * 0.99, 1_000_000)["action"] == "routine_monitoring"
    assert choose_action(be * 1.01, 1_000_000)["action"] != "routine_monitoring"
