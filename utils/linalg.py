
def unique_rows(a):

    # Unique Rows Code
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    unique_a = np.unique(b).view(a.dtype).reshape(-1, a.shape[1])
    return unique_a

def unique_rows_number(a):

	return (a[:, np.newaxis, :] == a).all(axis = 2).sum(axis = 1)
