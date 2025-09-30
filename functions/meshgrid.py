import torch 
def meshgrid (Wh, Ww):
    coords_h = list(range(Wh))  # row indices
    coords_w = list(range(Ww))  # col indices

    # Initialize empty grids
    grid_h = []
    grid_w = []

    # Fill grids manually
    for i in coords_h:   # loop over rows
        row_h = []
        row_w = []
        for j in coords_w:  # loop over cols
            row_h.append(i)  # row index
            row_w.append(j)  # col index
        grid_h.append(row_h)
        grid_w.append(row_w)

    # Stack along new "0th" dimension [2, Wh, Ww]
    stacked = [grid_h, grid_w]
    return stacked

Wh, Ww = 4, 4
coords_man = manual_meshgrid_stack(Wh, Ww)
# API (reference)
coords_h = torch.arange(Wh)
coords_w = torch.arange(Ww)
coords_api = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))

print ("The manual case")
print("Shape = [2, Wh, Ww] ->", len(coords_man), len(coords_man[0]), len(coords_man[0][0]))
print (coords_man)
print("\nAPI result:")
print(coords_api)
print("Shape =", coords_api.shape)

# Convert manual to tensor for comparison
coords_manual_tensor = torch.tensor(coords_man)
print("Manual result as a tensor:")
print(coords_manual_tensor)
print("Shape =", coords_manual_tensor.shape)