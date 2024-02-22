

def batch_iterator(lists, batch_size):
    """
    Yields batches of zipped lists.

    :param lists: A list of lists to be zipped and batched.
    :param batch_size: The size of each batch.
    """
    # Ensure all lists have the same length
    assert all(len(lst) == len(lists[0]) for lst in lists), "All lists must have the same length."

    # Calculate the number of batches
    total_size = len(lists[0])
    for i in range(0, total_size, batch_size):
        yield [list[i:i + batch_size] for list in lists]