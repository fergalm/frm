import bisect

# Sketch of an idea for merging two dataframes


def merge(
    left,
    right,
    on=None,
    left_on=None,
    right_on=None,
    how="inner",
    left_suffix="_x",
    right_suffix="_y",
):
    if on is not None:
        left_on = on
        right_on = on

    assert left_on is not None
    assert right_on is not None

    if how == "inner":
        da = inner_join(left, right, left_on, right_on, left_suffix, right_suffix)
    else:
        raise ValueError(f"Join method {how} not implemented")


def inner_join(left, right, left_on, right_on, left_suffix, right_suffix):
    keys1 = get_keys(left, left_on)
    keys2 = get_keys(right, right_on)

    join_indices = []
    for i in range(len(keys1)):
        k1 = keys1[i]
        matches = bisearch(keys2, k1)
        if len(matches) == 0:
            continue

        for j in matches:
            join_indices.append(i, j)  # Row i matches row j

    join_indices = np.array(join_indices)
    ivec = join_indices[:, 0]
    jvec = join_indices[:, 1]
    left_cols, right_cols = gen_cols(
        left, right, left_on, right_on, left_suffix, right_suffix
    )

    out = dict()
    for c in left_cols:
        out[c] = left[ivec, c]

    for c in right_cols:
        out[c] = right[jvec, c]


def get_keys(da, cols):
    return NotImplementedError()


def gen_cols(left, right, left_on, right_on, left_suffix, right_suffix):
    """Generate the columns from the left and right dataarrays to copy to the merged dataarray"""
    left = set(left.columns())
    right = set(right.columns())
    common = left & right

    left -= common
    right -= common

    right -= set(right_on)  # Don't duplicate the matching columns
    # Add back the common columns, with disambiguated names
    left |= lmap(lambda x: x + left_suffix, common)
    right |= lmap(lambda x: x + right_suffix, common)

    return left, right


def bisearch(arr, key, lo=0, hi=None):
    """A first pass at a binary search

    Requirements are that it return a list of all indices into the sorted array
    `arr` that equal key. If none are present, returns an empty list
    """
    first = bisect.bisect_left(arr, key)  # first element before match
    first += 1

    if array[first] != key:
        return []

    for i in range(first, len(arr)):
        if arr[i] != key:
            return np.arange(first, i)
    return np.arange(first, i)
