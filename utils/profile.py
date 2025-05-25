
def profile_model(model, inputs):
    from thop import profile
    from rich import print
    import sys
    macs, params = profile(model, inputs=inputs, verbose=False)
    print('Network initialized! (Input {} MB : MACs: {} M | Params: {} M)'.format(
        round(sys.getsizeof(inputs[0].untyped_storage())/1000/1000, 3),
        round(macs/1000/1000, 2), 
        round(params/1000/1000, 2),
    ))

    return macs, params

def get_optimal_worker_count(total_cores, num_workers_config=None, use_multiprocessing=False):
    """Determine the optimal number of workers for multiprocessing.
    
    Args:
        total_cores (int): Total number of CPU cores available
        num_workers_config (int, optional): Explicitly configured number of workers
        is_colab (bool): Whether running in Google Colab environment
    
    Returns:
        int: Optimal number of workers to use
    """
    if num_workers_config is not None and num_workers_config > 0:
        # If explicitly configured, use that number but cap at total_cores-1
        num_workers = min(num_workers_config, total_cores)
    else:
        # Default to total_cores - 1, optimized to even number for high core counts
        num_workers = 1 if not use_multiprocessing else 2 ** (total_cores.bit_length() - 1)
    
    return num_workers

def format_time(seconds: int) -> str:
    """Convert seconds to HH:MM:SS format if >= 1 hour, otherwise MM:SS format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"
