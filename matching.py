import numpy as np

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def match_players(f1, f2, threshold=50):
    matches = []
    for i, player1 in enumerate(f1):
        best = None
        dist = float('inf')
        for j, player2 in enumerate(f2):
            d = euclidean(player1["coords"], player2["coords"])
            if d < dist and d < threshold:
                dist = d
                best = j
        if best is not None:
            matches.append((i, best))
    return matches
