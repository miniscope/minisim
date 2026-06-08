"""Lightweight wall-time instrumentation for the ``minisim`` pipeline.

A :class:`PerfTracker` is an opt-in stopwatch you create and hand to
:func:`~minisim.simulate.simulate` or :func:`~minisim.video.simulate_video` via
their ``perf=`` argument. As the run proceeds, each instrumented region is timed
and accumulated by name, so you get a per-phase breakdown of where wall time
goes - which pipeline step dominates ``simulate()``, or how the streaming video
writer splits its time between rendering and encoding.

The design goals match the rest of the package: small, readable, and free when
unused. The orchestrators call the optional-aware :func:`measure` helper, which
is a no-op when ``perf is None`` (the default), so instrumentation adds no
measurable cost to an un-profiled run. When a tracker *is* supplied, spans of the
same name accumulate (summed duration and call count), so a phase that runs once
per chunk over many chunks reports as a single row with its total time and the
number of chunks - the natural granularity for both execution paths.

Typical use::

    from minisim import simulate
    from minisim.perf import PerfTracker

    perf = PerfTracker()
    rec = simulate(spec, perf=perf)
    print(perf)            # a per-phase table, slowest first

The tracker only records the spans the orchestrators open; it deliberately does
*not* hook into arbitrary functions. For a full call-graph profile (every
function, not just the named pipeline phases), reach for ``cProfile`` around a
``simulate`` call - this module is the cheap always-available overview, not a
replacement for a sampling/deterministic profiler.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter


@dataclass
class _SpanStat:
    """Accumulated timing for all spans sharing one name.

    ``domain`` is the optional pipeline domain (``cell`` / ``tissue`` / ``motion``
    / ``sensor``) of the step that opened the span, carried through purely so the
    report can show it; it is ``None`` for phases that are not a single step
    (e.g. the streaming render sub-phases, ``finalize``, ``encode+write``).
    """

    name: str
    domain: str | None = None
    seconds: float = 0.0
    calls: int = 0

    def add(self, dt: float) -> None:
        self.seconds += dt
        self.calls += 1


@dataclass
class PerfTracker:
    """Opt-in collector of named wall-time spans for one pipeline run.

    Pass an instance to ``simulate(spec, perf=...)`` or
    ``simulate_video(spec, path, perf=...)``; afterwards read :attr:`spans`,
    :meth:`total_seconds`, or just ``str(tracker)`` for a formatted table. A
    fresh tracker per run keeps the totals clean; reusing one across runs sums
    them (occasionally useful for an aggregate over a sweep).

    Spans are keyed by name and accumulate, so the report has one row per distinct
    phase regardless of how many times it ran. Insertion order is preserved, so the
    table can fall back to run order when two phases tie on time.
    """

    #: name -> accumulated stat, in first-seen order.
    spans: dict[str, _SpanStat] = field(default_factory=dict)

    @contextmanager
    def measure(self, name: str, *, domain: str | None = None) -> Iterator[None]:
        """Time the ``with`` block and add it to the span named ``name``.

        The duration is recorded even if the block raises, so a timed region that
        fails still shows the time it spent before failing.
        """
        start = perf_counter()
        try:
            yield
        finally:
            self._record(name, perf_counter() - start, domain)

    def _record(self, name: str, dt: float, domain: str | None) -> None:
        stat = self.spans.get(name)
        if stat is None:
            stat = self.spans[name] = _SpanStat(name, domain)
        stat.add(dt)

    def total_seconds(self) -> float:
        """Summed duration over every recorded span.

        Note these are wall-time spans that the orchestrators may nest (e.g. a
        per-step span inside an enclosing loop), so the total is the sum of the
        recorded regions, not necessarily the end-to-end run time. The
        orchestrators here open only non-overlapping spans, so in practice it
        tracks the instrumented portion of the run closely.
        """
        return sum(s.seconds for s in self.spans.values())

    def report(self) -> str:
        """A fixed-width table of phases sorted slowest-first.

        Columns: phase name, domain (blank when not a single step), call count,
        total seconds, and percent of tracked time. Returns a friendly note rather
        than an empty table when nothing was recorded (e.g. profiling a run that
        was cut short by ``until`` before any instrumented phase).
        """
        if not self.spans:
            return "PerfTracker: no spans recorded."
        total = self.total_seconds() or 1.0  # guard the percent divide for a 0s run
        rows = sorted(self.spans.values(), key=lambda s: s.seconds, reverse=True)
        name_w = max(len("phase"), *(len(s.name) for s in rows))
        dom_w = max(len("domain"), *(len(s.domain or "") for s in rows))
        header = (
            f"{'phase':<{name_w}}  {'domain':<{dom_w}}  "
            f"{'calls':>6}  {'seconds':>9}  {'%':>6}"
        )
        lines = [header, "-" * len(header)]
        for s in rows:
            lines.append(
                f"{s.name:<{name_w}}  {s.domain or '':<{dom_w}}  "
                f"{s.calls:>6d}  {s.seconds:>9.4f}  {100.0 * s.seconds / total:>5.1f}%"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'total':<{name_w}}  {'':<{dom_w}}  "
            f"{sum(s.calls for s in rows):>6d}  {self.total_seconds():>9.4f}  {100.0:>5.1f}%"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.report()


@contextmanager
def measure(
    perf: PerfTracker | None, name: str, *, domain: str | None = None
) -> Iterator[None]:
    """Optional-aware span: a no-op when ``perf is None``, else ``perf.measure``.

    This is the form the orchestrators call so a span can be opened unconditionally
    at the call site while staying free for the common un-profiled run. The
    ``perf is None`` fast path skips even the ``perf_counter`` calls.
    """
    if perf is None:
        yield
        return
    with perf.measure(name, domain=domain):
        yield
