r"""The implementation of hpsg parsing with O(n^5) time complexity. """

def split_calculate(h, i, j, s, dep, left=True):
    max_score_r, max_split_r, correspond_k = float('-inf'), None, None

    # find the best sub-span head word r
    r_select = range(i, h) if left else range(h+1, j+1)
    for r in r_select:
        max_score_k, max_split_k = float('-inf'), None

        # find the best split k
        k_select = range(r, h) if left else range(h, r)
        for k in k_select:
            score_k = s[i, k, r] + s[k+1, j, h] if left \
                else s[i, k, h] + s[k+1, j, r]
            max_score_k, max_split_k = (score_k, k) if score_k > max_score_k \
                else (max_score_k, max_split_k)
        score_r = max_score_k + dep[r, h]

        if score_r > max_score_r:
            max_score_r = score_r
            max_split_r = r
            correspond_k = max_split_k

    return max_score_r, max_split_r, correspond_k

def split_search(h, i, j, s, dep):
    left, rl, kl = split_calculate(h, i, j, s, dep, left=True)
    right, rr, kr = split_calculate(h, i, j, s, dep, left=False)
    return (left, rl, kl) if left >= right else (right, rr, kr)

def root_calculate(s, n, dep, root_pos=0):
    score_tot, root = float('-inf'), None
    for h in range(1, n+1):
        score = s[1, n, h] + dep[h, root_pos]
        if score > score_tot:
            score_tot, root = score, h
    assert root is not None
    return score_tot, root

def joint_span_parsing(s_dep, s_con, lens):
    
    def decode(dep, con, n, heads):
        s = con.new_zeros((n+1, n+1, n+1))
        p_r = con.new_zeros((n+1, n+1, n+1)).int()
        p_k = con.new_zeros((n+1, n+1, n+1)).int()

        for length in range(1, n+1):
            for i in range(1, n-length+2):
                j = i + length - 1
                if length == 1:
                    s[i, j, i] = con[i-1, j]
                else:
                    for h in range(i, j+1):
                        max_split, r, k = split_search(h, i, j, s, dep)
                        s[i, j, h] = max_split + con[i-1, j]
                        p_r[i, j, h] = r
                        p_k[i, j, h] = k
        _, root = root_calculate(s, n, dep)
        tree = track(p_r, p_k, 1, n, root, heads)
        return heads, tree

    def track(p_r, p_k, i, j, h, heads):
        if i == j:
            return [[int(i-1), int(i)]]
        else:
            sub_head = p_r[i, j, h]
            split = p_k[i, j, h]
            heads[sub_head] = h
        left_head, right_head = sorted((h, sub_head))
        ltree = track(p_r, p_k, i, split, left_head, heads)
        rtree = track(p_r, p_k, split+1, j, right_head, heads)
        return [[int(i)-1, int(j)]] + ltree + rtree

    batch_size, seq_len, _ = s_dep.shape
    arcs = s_dep.new_zeros((batch_size, seq_len)).long()
    trees = []
    for i in range(batch_size):
        arc, tree = decode(s_dep[i], s_con[i], lens[i], arcs[i])
        arcs[i] = arc
        trees.append(tree)

    return arcs, trees

