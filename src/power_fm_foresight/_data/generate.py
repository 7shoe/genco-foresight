"""The original episode assignment and v4 demand law."""
import numpy as np

SPLITS = (("train", 12, 10), ("model_selection", 2, 2), ("risk_calibration", 2, 2), ("test", 4, 3))

def episode_plan(seed=42, source_blocks_one_based=None, split_episodes=None):
    """Disjoint 21-day source blocks; reuse is restricted to the same partition."""
    blocks = np.random.default_rng(seed).permutation(17)
    counts = split_episodes or {split: count for split, count, _ in SPLITS}
    if set(counts) != {s for s, _, _ in SPLITS} or sum(counts.values()) != 20:
        raise ValueError("Episode partitions must assign all 20 episodes")
    if source_blocks_one_based is not None:
        expected = {split for split, _, _ in SPLITS}
        flattened = [i for values in source_blocks_one_based.values() for i in values]
        if (set(source_blocks_one_based) != expected or
                any(type(i) is not int for i in flattened) or sorted(flattened) != list(range(1, 18))):
            raise ValueError("Source pools must be disjoint and cover blocks 1 through 17 exactly once")
        if any(not values or len(values) > counts[split] for split, values in source_blocks_one_based.items()):
            raise ValueError("Each pool must be nonempty and covered by its assigned episodes")
    episodes, pools, offset = [], {}, 0
    for split, count, pool_count in SPLITS:
        pool = (sorted(i-1 for i in source_blocks_one_based[split]) if source_blocks_one_based is not None
                else sorted(int(i) for i in blocks[offset:offset+pool_count]))
        pools[split] = pool
        offset += pool_count
        for j in range(counts[split]):
            # Cover every source block before reusing one, with independent noise.
            block = pool[j % len(pool)]
            episodes.append({"episode_id": f"{split}_{j:02d}", "split": split,
                "source_block": block, "source_start_hour": block*504, "hours": 504,
                "seed": int(np.random.SeedSequence([seed, len(episodes)]).generate_state(1)[0])})
    return episodes, pools



def demands(profile, baseline_pd, baseline_qd, settings, seed):
    """Datakit's multiplicative uniform law plus stationary lognormal AR(1)."""
    rho, innovation, sigma = settings["rho"], settings["innovation_std"], settings["sigma"]
    if not abs(rho) < 1 or not 0 <= sigma < 1 or innovation < 0 or not (np.asarray(profile) > 0).all():
        raise ValueError("Invalid positive-load sampling law")
    rng = np.random.default_rng(seed)
    variance = innovation**2/(1-rho**2)
    u = rng.normal(0, np.sqrt(variance), len(baseline_pd))
    pd_values = []
    for ref in profile:
        pd_values.append(baseline_pd * ref * rng.uniform(1-sigma, 1+sigma, len(u)) * np.exp(u-variance/2))
        u = rho*u+rng.normal(0, innovation, len(u))
    pd_values = np.asarray(pd_values)
    qd_values = np.broadcast_to(baseline_qd, pd_values.shape).copy()
    active = baseline_pd != 0
    qd_values[:, active] = pd_values[:, active] * (baseline_qd[active]/baseline_pd[active])
    return pd_values, qd_values
