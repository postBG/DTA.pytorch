def linear_rampup(current, rampup_length):
    """Linear rampup by myself"""
    return min(1.0, (current + 1) / rampup_length)
