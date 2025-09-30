# -----------------------------
# 1) Convert tensor -> nested lists
#    (innermost is a list of channels)
# -----------------------------
x_list = x_api.tolist()  # nested python lists: [B][H][W][C]

# -----------------------------
# 2) Helper: extract patches from nested-list with same layout as API slicing
# -----------------------------
def extract_patch_list(x_list, start_h, start_w):
    B = len(x_list)
    H = len(x_list[0])
    W = len(x_list[0][0])
    C = len(x_list[0][0][0])
    outH = H // 2
    outW = W // 2
    # preallocate [B][outH][outW][C]
    out = [[[ [0]*C for _ in range(outW)] for _ in range(outH)] for _ in range(B)]
    for b in range(B):
        for i in range(start_h, H, 2):
            oi = i // 2
            for j in range(start_w, W, 2):
                oj = j // 2
                # x_list[b][i][j] is a list of length C (channels)
                out[b][oi][oj] = x_list[b][i][j]
    return out

x0_n = extract_patch_list(x_list, 0, 0)  # top-left
x1_n = extract_patch_list(x_list, 1, 0)  # bottom-left
x2_n = extract_patch_list(x_list, 0, 1)  # top-right
x3_n = extract_patch_list(x_list, 1, 1)  # bottom-right

# -----------------------------
# 3) Manual concat along last dim (pure indexing, no append/extend)
# -----------------------------
def manual_cat_lastdim_pure(tensors):
    # tensors: list of nested lists each shaped [B][H2][W2][C]
    B = len(tensors[0])
    H2 = len(tensors[0][0])
    W2 = len(tensors[0][0][0])
    C = len(tensors[0][0][0][0])
    T = len(tensors)
    outC = C * T
    # allocate result [B][H2][W2][outC]
    res = [[[ [0]*outC for _ in range(W2)] for _ in range(H2)] for _ in range(B)]
    for b in range(B):
        for h in range(H2):
            for w in range(W2):
                offset = 0
                for t_idx in range(T):
                    ch_list = tensors[t_idx][b][h][w]
                    for c in range(len(ch_list)):
                        res[b][h][w][offset + c] = ch_list[c]
                    offset += len(ch_list)
    return res

manual_res = manual_cat_lastdim_pure([x0_n, x1_n, x2_n, x3_n])

