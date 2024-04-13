from functools import partial

# Define functions for processing data
def add_5(x):
    return x + 5

def multiply_by_2(x):
    return x * 2

def subtract_3(x):
    return x - 3

# Pipeline using the pipeline operator
result = (10 | partial(add_5) | partial(multiply_by_2) | partial(subtract_3))
print(result)  # Output: 27
