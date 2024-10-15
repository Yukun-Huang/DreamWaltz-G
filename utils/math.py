

def normalized_cross_correlation(x, y):
    std_x, mean_x = torch.std_mean(x)
    std_y, mean_y = torch.std_mean(y)
    return torch.mean((x - mean_x) * (y - mean_y)) / (std_x * std_y)
