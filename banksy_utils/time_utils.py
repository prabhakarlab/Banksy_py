"""
Decorator to measure and display execution time of a function

Nigel 3 dec 2020
"""

import time

def timer(func):
    
    def _timer(*args, **kwargs):
        
        start_time = time.time()
        
        try:
            return func(*args, **kwargs)
        
        finally:
            time_taken = time.time() - start_time
            print(f"---- Ran {func.__name__} "
                  f"in {time_taken:0.2f} s ----\n")

        return result
    
    return _timer
