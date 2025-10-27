def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def compute_experience_match(candidate_code: int,
                             job_code: int,
                             num_levels: int,
                             mode: str = 'combined',
                             candidate_weight: float = 0.5,
                             bias: float = 0.25) -> float:
    """Compute an experience match score in [0,1].

    Modes:
      - 'candidate_only': score depends only on candidate seniority (more senior => higher)
      - 'job_only': score depends only on symmetric distance between candidate and job
      - 'combined': weighted avg of candidate_only and job_only (candidate_weight controls mix)

    Parameters:
      candidate_code: integer encoding (0..max)
      job_code: integer encoding (0..max) — may be ignored depending on mode
      num_levels: total number of experience levels (e.g., 3)
      candidate_weight: weight given to candidate seniority when mode='combined'
      bias: reserved for future use (kept for API compatibility)

    Returns:
      float in [0,1]
    """
    if num_levels <= 0:
        return 1.0

    max_code = num_levels - 1

    # Candidate-only: normalize candidate_code to [0,1]
    if max_code > 0:
        candidate_only = candidate_code / max_code
    else:
        candidate_only = 1.0

    # Job-only (symmetric distance-based)
    job_only = 1.0 - abs(candidate_code - job_code) / num_levels

    if mode == 'candidate_only':
        return clamp(candidate_only)
    if mode == 'job_only':
        return clamp(job_only)

    # Default: combined
    cw = clamp(candidate_weight, 0.0, 1.0)
    combined = cw * candidate_only + (1.0 - cw) * job_only
    return clamp(combined)
