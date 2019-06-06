def get_index_from_y(l, y):
    i = 0
    if i >= len(l):
        return i
    while y < l[i].y:
        i += 1
        if i >= len(l):
            break
    return i