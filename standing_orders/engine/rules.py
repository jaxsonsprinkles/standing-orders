"""T3 time logic. Deterministic: same inputs and clock give the same output.

A rule is parsed once into a tree of nodes. Each node holds its own time state and returns a Tri on
every update. Kleene logic throughout: UNKNOWN never produces TRUE, so it can never cause a firing.

Operator shapes:
  "pred_name"
  {"all": [rule, ...]}  {"any": [rule, ...]}  {"not": rule}
  {"held_for": {"seconds": N, "of": rule}}   TRUE once `of` has been TRUE continuously for N s
  {"rose": rule}                              TRUE on the update where `rule` goes FALSE -> TRUE
  {"fell": rule}                              TRUE on the update where `rule` goes TRUE -> FALSE
  {"within": {"seconds": N, "after": rule_a, "then": rule_b}}
                                              TRUE while `then` is TRUE within N s of `after` rising
"""

from __future__ import annotations
import math

from .clock import Clock
from .program import ProgramError, Rule
from .types import Tri

T, F, U = Tri.TRUE, Tri.FALSE, Tri.UNKNOWN


class _Node:
    def eval(self, values: dict[str, Tri], now: float) -> Tri:
        raise NotImplementedError


class _Pred(_Node):
    def __init__(self, name: str):
        self.name = name

    def eval(self, values, now):
        return values.get(self.name, U)


class _All(_Node):
    def __init__(self, kids):
        self.kids = kids

    def eval(self, values, now):
        vs = [k.eval(values, now) for k in self.kids]  # evaluate all: children keep time state
        return F if F in vs else U if U in vs else T


class _Any(_Node):
    def __init__(self, kids):
        self.kids = kids

    def eval(self, values, now):
        vs = [k.eval(values, now) for k in self.kids]
        return T if T in vs else U if U in vs else F


class _Not(_Node):
    def __init__(self, kid):
        self.kid = kid

    def eval(self, values, now):
        v = self.kid.eval(values, now)
        return U if v is U else F if v is T else T


class _HeldFor(_Node):
    def __init__(self, seconds: float, kid):
        self.seconds, self.kid = seconds, kid
        self.since: float | None = None

    def eval(self, values, now):
        v = self.kid.eval(values, now)
        if v is not T:
            self.since = None  # an unknown gap breaks "continuously", so it resets too
            return v
        if self.since is None:
            self.since = now
        return T if now - self.since >= self.seconds else F


class _Edge(_Node):
    """rose: FALSE -> TRUE. fell: TRUE -> FALSE. UNKNOWN is never a valid edge endpoint."""

    def __init__(self, kid, before: Tri, after: Tri):
        self.kid, self.before, self.after = kid, before, after
        self.prev: Tri | None = None

    def eval(self, values, now):
        v = self.kid.eval(values, now)
        prev, self.prev = self.prev, v
        if v is U:
            return U
        return T if (prev is self.before and v is self.after) else F


class _Within(_Node):
    def __init__(self, seconds: float, a, b):
        self.seconds, self.start = seconds, None
        self.a = _Edge(a, F, T)
        self.b = b

    def eval(self, values, now):
        if self.a.eval(values, now) is T:
            self.start = now
        b = self.b.eval(values, now)
        if self.start is None or now - self.start > self.seconds:
            return F
        return b


def _seconds(body: dict, op: str) -> float:
    s = body.get("seconds")
    if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s) or s <= 0:
        raise ProgramError(f"{op}: 'seconds' must be a positive number, got {s!r}")
    return float(s)


def _need(body, keys: tuple[str, ...], op: str) -> dict:
    if not isinstance(body, dict) or set(body) != {"seconds", *keys}:
        raise ProgramError(f"{op}: expected keys {sorted(('seconds', *keys))}, got {body!r}")
    return body


def compile_rule(rule: Rule) -> _Node:
    """Parse a rule into nodes. Raises ProgramError naming the offending operator or shape."""
    if isinstance(rule, str):
        return _Pred(rule)
    if not isinstance(rule, dict) or len(rule) != 1:
        raise ProgramError(f"a rule must be a predicate name or a one-key operator object, got {rule!r}")
    (op, body), = rule.items()
    if op in ("all", "any"):
        if not isinstance(body, list) or not body:
            raise ProgramError(f"{op}: expected a non-empty list of rules")
        return (_All if op == "all" else _Any)([compile_rule(r) for r in body])
    if op == "not":
        return _Not(compile_rule(body))
    if op == "rose":
        return _Edge(compile_rule(body), F, T)
    if op == "fell":
        return _Edge(compile_rule(body), T, F)
    if op == "held_for":
        body = _need(body, ("of",), op)
        return _HeldFor(_seconds(body, op), compile_rule(body["of"]))
    if op == "within":
        body = _need(body, ("after", "then"), op)
        return _Within(_seconds(body, op), compile_rule(body["after"]), compile_rule(body["then"]))
    raise ProgramError(f"unknown operator {op!r}")


def referenced_predicates(rule: Rule) -> set[str]:
    """Every predicate name the rule mentions. Assumes the rule already compiles."""
    if isinstance(rule, str):
        return {rule}
    (_, body), = rule.items()
    if isinstance(body, list):
        kids = body
    elif isinstance(body, dict):
        kids = [v for k, v in body.items() if k != "seconds"]
    else:
        kids = [body]
    return set().union(*(referenced_predicates(k) for k in kids))


class RuleEvaluator:
    def __init__(self, rule: Rule, clock: Clock, hysteresis_seconds: float = 0.0):
        """hysteresis_seconds: a predicate must hold a new TRUE/FALSE value this long before the
        rules see it, which smooths flicker. UNKNOWN passes through at once (the safe direction)."""
        self.rule, self.clock = rule, clock
        self._root = compile_rule(rule)
        self._hyst = hysteresis_seconds
        self._stable: dict[str, Tri] = {}
        self._pending: dict[str, tuple[Tri, float]] = {}

    def _smooth(self, values: dict[str, Tri], now: float) -> dict[str, Tri]:
        if self._hyst <= 0:
            return values
        out = {}
        for name, v in values.items():
            if v is U or name not in self._stable:
                self._stable[name] = v
                self._pending.pop(name, None)
            elif v is not self._stable[name]:
                cand, since = self._pending.get(name, (v, now))
                if cand is not v:
                    cand, since = v, now
                self._pending[name] = (cand, since)
                if now - since >= self._hyst:
                    self._stable[name] = v
                    del self._pending[name]
            else:
                self._pending.pop(name, None)
            out[name] = self._stable[name]
        return out

    def update(self, values: dict[str, Tri]) -> Tri:
        """Feed the latest predicate values; TRUE means the rule is satisfied now."""
        now = self.clock.now()
        return self._root.eval(self._smooth(values, now), now)
