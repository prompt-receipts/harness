"""Stdlib-only statistics for comparing two pass rates."""
import math
import random


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _hyper_pmf(k, K, n, N):
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def fisher_exact(a_pass, a_n, b_pass, b_n):
    """Two-sided Fisher exact p-value (sum of tables no more likely than observed)."""
    K = a_pass + b_pass
    N = a_n + b_n
    obs = _hyper_pmf(a_pass, K, a_n, N)
    lo, hi = max(0, K - b_n), min(K, a_n)
    p = sum(
        pr for k in range(lo, hi + 1)
        if (pr := _hyper_pmf(k, K, a_n, N)) <= obs * (1 + 1e-9)
    )
    return min(1.0, p)


def cluster_bootstrap_diff(per_task, iters=5000, seed=0):
    """per_task: {task: {"A": [0/1...], "B": [0/1...]}}. Resamples whole tasks.
    Returns (diff_B_minus_A, lo, hi) as a 95% percentile interval."""
    tasks = sorted(per_task)

    def diff(sample):
        a = [x for t in sample for x in per_task[t]["A"]]
        b = [x for t in sample for x in per_task[t]["B"]]
        return sum(b) / len(b) - sum(a) / len(a)

    point = diff(tasks)
    rng = random.Random(seed)
    ds = sorted(diff([rng.choice(tasks) for _ in tasks]) for _ in range(iters))
    return point, ds[int(0.025 * iters)], ds[int(0.975 * iters) - 1]
