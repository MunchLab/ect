def next_vert_name(s, num_verts=1):
    """Generate sequential vertex names (alphabetical or numerical)."""
    if isinstance(s, int):
        return [s + i + 1 for i in range(num_verts)] if num_verts > 1 else s + 1

    def increment_char(c):
        return "A" if c == "Z" else chr(ord(c) + 1)

    def increment_str(s):
        chars = list(s)
        for i in reversed(range(len(chars))):
            chars[i] = increment_char(chars[i])
            if chars[i] != "A":
                break
            elif i == 0:
                return "A" + "".join(chars)
        return "".join(chars)

    # handle multiple increments
    names = [s]
    for _ in range(num_verts):
        names.append(increment_str(names[-1]))
    return names[1:] if num_verts > 1 else names[1]
