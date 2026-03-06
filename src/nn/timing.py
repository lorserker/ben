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
        return False


def timed_inference(model_name: str):
    """Decorator for timing model inference methods."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ModelTimer.time_call(model_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
