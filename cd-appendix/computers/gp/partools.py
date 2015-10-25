"""
Parallel Programming Wrappers
"""
import parmap

def parallel_map(*args, processes = 1):
    """
    Wrapper function for 'parmap.map': Parallises the computations in 
    'map' form if required. If only one process is needed, computations 
    are performed serially
    """
    if processes == 1:
        return [args[0](element, *args[2:]) for element in args[1]]
    return parmap.map(*args, processes = processes)

def parallel_starmap(f, args, processes = 1):
    """
    Wrapper function for 'parmap.starmap': Parallises the computations in 
    'starmap' form if required. If only one process is needed, computations 
    are performed serially
    """
    if processes == 1:
        return [f(*arg) for arg in args]
    return parmap.starmap(f, args, processes = processes)

def set_log_level(level):
    """
    Sets the logging level for tasks operating in parallel on different cores

    Arguments:
        level:  logging levels from the logging module
    Returns:
        None
    """
    parmap.multiprocessing.get_logger().setLevel(level)

def get_log_level():
    """
    Gets the logging level for tasks operating in parallel on different cores

    Arguments:
        None
    ReturnsL
        level:  logging levels from the logging module
    """
    parmap.multiprocessing.get_logger().getEffectiveLevel()