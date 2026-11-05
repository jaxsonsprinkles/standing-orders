"""T3 time logic. Deterministic: same inputs and clock give the same output."""
from .clock import Clock
from .program import Rule
from .types import Tri


class RuleEvaluator:
    def __init__(self, rule: Rule, clock: Clock):
        self.rule, self.clock = rule, clock

    def update(self, values: dict[str, Tri]) -> Tri:
        """Feed the latest predicate values; TRUE means the rule is satisfied now. Day 2."""
        raise NotImplementedError
