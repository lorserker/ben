"""
Timing utilities for neural network model inference.

Usage:
    from nn.timing import ModelTimer

    # Enable timing globally
    ModelTimer.enabled = True

    # Get timing stats
    print(ModelTimer.get_summary())

    # Reset stats
    ModelTimer.reset()
"""

import functools
import time
from collections import defaultdict
from threading import Lock


class ModelTimer:
    """Thread-safe singleton for tracking model inference times."""

    enabled = False  # Set to True to enable timing
    _stats = defaultdict(lambda: {'count': 0, 'total_ms': 0.0, 'min_ms': float('inf'), 'max_ms': 0.0, 'items': 0})
    _lock = Lock()

    @classmethod
    def record(cls, model_name: str, duration_ms: float, items: int = 0):
        """Record a timing measurement for a model.

        Args:
            model_name: Name of the model/operation
            duration_ms: Duration in milliseconds
            items: Optional number of items processed (e.g. boards solved)
        """
        if not cls.enabled:
            return

        with cls._lock:
            stats = cls._stats[model_name]
            stats['count'] += 1
            stats['total_ms'] += duration_ms
            stats['min_ms'] = min(stats['min_ms'], duration_ms)
            stats['max_ms'] = max(stats['max_ms'], duration_ms)
            stats['items'] += items

    @classmethod
    def get_stats(cls, model_name: str = None):
        """Get timing stats for a specific model or all models."""
        with cls._lock:
            if model_name:
                return dict(cls._stats.get(model_name, {}))
            return {k: dict(v) for k, v in cls._stats.items()}

    @classmethod
    def get_summary(cls):
        """Get a formatted summary of all timing stats."""
        with cls._lock:
            # Check if any entry has items tracked
            has_items = any(s['items'] > 0 for s in cls._stats.values())

            lines = []
            width = 92 if has_items else 70
            lines.append("=" * width)
            lines.append("MODEL TIMING SUMMARY")
            lines.append("=" * width)
            header = f"{'Model':<30} {'Count':>8} {'Total(ms)':>12} {'Avg(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
            if has_items:
                header += f" {'Items':>10} {'Items/s':>10}"
            lines.append(header)
            lines.append("-" * width)

            total_count = 0
            total_time = 0.0

            for model_name in sorted(cls._stats.keys()):
                stats = cls._stats[model_name]
                count = stats['count']
                total_ms = stats['total_ms']
                avg_ms = total_ms / count if count > 0 else 0
                min_ms = stats['min_ms'] if count > 0 else 0
                max_ms = stats['max_ms']
                items = stats['items']

                total_count += count
                total_time += total_ms

                line = f"{model_name:<30} {count:>8} {total_ms:>12.2f} {avg_ms:>10.2f} {min_ms:>10.2f} {max_ms:>10.2f}"
                if has_items:
                    if items > 0:
                        throughput = items / (total_ms / 1000) if total_ms > 0 else 0
                        line += f" {items:>10} {throughput:>10.1f}"
                    else:
                        line += f" {'':>10} {'':>10}"
                lines.append(line)

            lines.append("-" * width)
            lines.append(f"{'TOTAL':<30} {total_count:>8} {total_time:>12.2f}")
            lines.append("=" * width)

            # Explain the dds_solve_* rows if any are present. The label encodes
            # both the trick (tNN) and the *purpose* of the DDS batch, so the
            # same trick can legitimately show several rows (e.g. card-play
            # evaluation plus one or more claim checks).
            if any(k.startswith('dds_solve_t') for k in cls._stats):
                lines.append("dds_solve_tNN_<purpose> = double-dummy solve at trick NN. Count = batches; Items = boards solved.")
                lines.append("  purpose: play=card-play eval | claimcheck=is a claim safe | claimtricks=max claimable")
                lines.append("           lead=opening-lead eval | bid=expected tricks during bidding | contract=bad-contract check")
                lines.append("  Several rows for one trick just mean DDS ran for different purposes there (e.g. play + claim check).")
                lines.append("=" * width)

            return "\n".join(lines)

    @classmethod
    def reset(cls):
        """Reset all timing stats."""
        with cls._lock:
            cls._stats.clear()

    @classmethod
    def time_call(cls, model_name: str, items: int = 0):
        """Context manager for timing a model call.

        Args:
            model_name: Name of the model/operation
            items: Number of items processed (e.g. boards solved)
        """
        return _TimingContext(model_name, items)


class _TimingContext:
    """Context manager for timing model calls."""

    def __init__(self, model_name: str, items: int = 0):
        self.model_name = model_name
        self.items = items
        self.start_time = None

    def __enter__(self):
        if ModelTimer.enabled:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if ModelTimer.enabled and self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            ModelTimer.record(self.model_name, duration_ms, self.items)
            # [robot-time] a neural network call also counts toward THIS decision's nn figure.
            if self.model_name.startswith(_NN_LABELS):
                PlayClock.add_nn(duration_ms / 1000.0)
        return False


def timed_inference(model_name: str):
    """Decorator for timing model inference methods."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ModelTimer.time_call(model_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# [ben-time] PER-DECISION STOPWATCH
#
# ModelTimer above accumulates across a whole run, which answers "where does a
# session spend its time". PlayClock answers the other question - "where did
# THIS card go" - because that is the one a slow table needs: a card that took
# four seconds has spent them in sampling, in double dummy or in PIMC, and the
# three are tuned by different settings.
#
# It is thread-local, so two decisions in flight cannot add to each other's
# total, and it is unconditional: a few float additions and one dict per card,
# against a decision that costs hundreds of milliseconds. Nothing is kept
# between cards - start() throws the previous one away.
# ---------------------------------------------------------------------------

import threading

# The ModelTimer labels that are a neural network being run - every wrapper under nn/ uses one
# of these. Double dummy (dds_*), PIMC, ACE and 'sampling_play' are stages, not networks.
_NN_LABELS = ('player_', 'bidder', 'bidinfo', 'contract', 'trick', 'leader', 'single_dummy')

class PlayClock:
    """Milliseconds spent in each stage of ONE card decision."""

    _local = threading.local()

    @classmethod
    def start(cls):
        cls._local.d = {'sampling': 0.0, 'dd': 0.0, 'pimc': 0.0, 'playouts': 0}

    @classmethod
    def add(cls, stage, seconds):
        d = getattr(cls._local, 'd', None)
        if d is not None:
            d[stage] = d.get(stage, 0.0) + seconds * 1000.0

    @classmethod
    def add_playouts(cls, n):
        # PIMC retries itself without constraints when it finds nothing, so this
        # accumulates rather than overwrites: the number reported is the playouts
        # the decision as a whole completed.
        d = getattr(cls._local, 'd', None)
        if d is not None and n is not None:
            try:
                d['playouts'] = d.get('playouts', 0) + int(n)
            except Exception:
                pass

    @classmethod
    def add_nn(cls, seconds):
        """
        [robot-time] Time inside a neural network. Most of it happens INSIDE a stage - the
        sampler scores its deals with the bidding and play networks - so it is also kept per
        stage (nn_in_sampling, nn_in_dd, nn_in_pimc). That is what lets the line report
        sampling and double dummy WITHOUT their network time, and the four parts add up.
        """
        d = getattr(cls._local, 'd', None)
        if d is None:
            return
        try:
            ms = seconds * 1000.0
            d['nn'] = d.get('nn', 0.0) + ms
            inside = d.get('_in')
            if inside:
                d['nn_in_' + inside] = d.get('nn_in_' + inside, 0.0) + ms
        except Exception:
            pass

    @classmethod
    def add_solved(cls, seconds, deals, pool_threads, os_threads):
        """
        [robot-time] One double-dummy batch: the time in the solver alone, how many deals it
        solved, the size of the solver's pool and how many OPERATING SYSTEM threads actually
        did the solving. The last two differ if the pool's threads are not real ones.
        """
        d = getattr(cls._local, 'd', None)
        if d is None:
            return
        try:
            d['solve'] = d.get('solve', 0.0) + seconds * 1000.0
            d['deals'] = d.get('deals', 0) + int(deals)
            d['threads'] = int(pool_threads)
            d['os_threads'] = max(d.get('os_threads', 0), int(os_threads))
        except Exception:
            pass

    @classmethod
    def enter(cls, name):
        """The outermost stage in progress, for add_nn. Returns True if this call set it."""
        d = getattr(cls._local, 'd', None)
        if d is None or d.get('_in'):
            return False
        d['_in'] = name
        return True

    @classmethod
    def leave(cls, mine):
        d = getattr(cls._local, 'd', None)
        if mine and d is not None:
            d['_in'] = None

    @classmethod
    def get(cls):
        return getattr(cls._local, 'd', None) or {'sampling': 0.0, 'dd': 0.0, 'pimc': 0.0, 'playouts': 0}

    @classmethod
    def stage(cls, name):
        return _PlayStage(name)


class _PlayStage:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()
        self.mine = PlayClock.enter(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        PlayClock.leave(self.mine)
        PlayClock.add(self.name, time.perf_counter() - self.t)
        return False


def timed_stage(name):
    """Decorator form of PlayClock.stage, for a method with several returns."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t = time.perf_counter()
            mine = PlayClock.enter(name)
            try:
                return func(*args, **kwargs)
            finally:
                PlayClock.leave(mine)
                PlayClock.add(name, time.perf_counter() - t)
        return wrapper
    return decorator
