def expand_ipv6(ipv6_short):
    '''
    Expand a shortened IPv6 address to its full representation.
    '''
    blocks = ipv6_short.split("::")
    if len(blocks) == 1:
        return ipv6_short
    first_half = blocks[0].split(":")
    second_half = blocks[1].split(":")
    num_blocks = 8 - (len(first_half) + len(second_half))
    expanded_address = ":".join(first_half + ["0"] * num_blocks + second_half)
    return expanded_address
