
def profile_model(model, inputs):
    from thop import profile
    from rich import print
    import sys
    macs, params = profile(model, inputs=inputs, verbose=False)
    print('모델 생성 완료! (Input {}: MACs: {} M | Params: {} K)'.format(
        sys.getsizeof(inputs[0].storage()),
        round(macs/1000/1000, 2), 
        round(params/1000, 2),
    ))

def get_optimal_worker_count(total_cores, num_workers_config=None):
    """Determine the optimal number of workers for multiprocessing.
    
    Args:
        total_cores (int): Total number of CPU cores available
        num_workers_config (int, optional): Explicitly configured number of workers
        is_colab (bool): Whether running in Google Colab environment
    
    Returns:
        int: Optimal number of workers to use
    """
    def _get_optimal_count(target_count):
        """Get the closest power of 2 or even number that's less than or equal to target_count."""
        # For high core counts (>32), prefer even numbers over powers of 2
        if target_count > 32:
            return (target_count // 2) * 2  # Round down to nearest even number
        
        # For lower core counts, prefer powers of 2
        power_of_2 = 1
        while power_of_2 * 2 <= target_count:
            power_of_2 *= 2
        
        # If we're close to the next power of 2, use that instead
        if target_count - power_of_2 < power_of_2 * 0.25:  # If within 25% of next power of 2
            return power_of_2
        # Otherwise use the closest even number
        return (target_count // 2) * 2

    if num_workers_config is not None and num_workers_config > 0:
        # If explicitly configured, use that number but cap at total_cores-1
        num_workers = min(num_workers_config, total_cores - 1)
    else:
        # Default to total_cores - 1, optimized to even number for high core counts
        num_workers = _get_optimal_count(total_cores - 1)
    
    return num_workers
