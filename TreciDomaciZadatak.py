# Domaci zadatak sam uradio koriscenjem programskog jezika Python.
# Na predavanju ste spomenuli da za izradu domaceg zadatka mozemo koristiti NetworkX biblioteku, i vestacku inteligenciju.
# Za modelovanje grafa koristio sam biblioteku NetworkX, konkretno klasa MultiGraph, kako bi se pravilno obradile paralelne grane u grafu.
# Za pojedine proracune (setnje duzine k) koristio sam matrice susedstva i linearnu algebru (NumPy).
# U izradi i proveri funkcionalnosti koda koristio sam alate vestacke inteligencije.
# Graf se tretira kao neusmeren multigraf.
# Lista susedstva je simetricna, time je omoguceno pravilno racunanje stepena i broja setnji uz ocuvanje paralelnih grana.

import networkx as nx
import numpy as np
from collections import Counter, deque

# Ulaz v i t:
v, t = 10, 2

# Lista susedstva:
adj_text = """
0:1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 12, 13, 17, 19, 19, 19, 21, 31
1:0, 2, 2, 3, 7, 13, 17, 19, 21, 30
2:0, 1, 1, 3, 3, 3, 7, 8, 9, 13, 27, 27, 27, 27, 28, 32
3:0, 1, 2, 2, 2, 7, 12, 13
4:0, 6, 10
5:0, 6, 10, 16
6:0, 4, 5, 16
7:0, 1, 2, 3
8:0, 2, 30, 32, 33
9:2, 33
10:0, 4, 5
11:0
12:0, 0, 3
13:0, 1, 2, 3, 33
14:32, 33
15:32, 33
16:5, 6
17:0, 1
18:32, 33
19:0, 0, 0, 1, 33
20:32, 32, 32, 33, 33, 33, 33
21:0, 1
22:32, 33
23:25, 27, 29, 32, 33
24:25, 27, 31
25:23, 24, 31
26:29, 33, 33, 33
27:2, 2, 2, 2, 23, 24, 33
28:2, 31, 33
29:23, 26, 32, 32, 32, 32, 33
30:1, 8, 32, 33
31:0, 24, 25, 28, 32, 33
32:2, 8, 14, 15, 18, 20, 20, 20, 22, 23, 29, 29, 29, 29, 30, 31, 33
33:8, 9, 13, 14, 15, 18, 19, 20, 20, 20, 20, 22, 23, 26, 26, 26, 27, 28, 29, 30, 31, 32
""".strip()


# Pretvaranje liste susedstva u strukturu podataka odnosno obrada unetog teksta u upotrebljiv oblik.
def parse_adj(text: str) -> dict[int, list[int]]:
    adj = {}
    for ln in (l.strip() for l in text.splitlines() if l.strip()):
        left, right = ln.split(":", 1)
        u = int(left.strip())
        nbrs = [int(x.strip()) for x in right.split(",") if x.strip()] if right.strip() else []
        adj[u] = nbrs

    all_nodes = set(adj.keys())
    for u, nbrs in adj.items():
        all_nodes.update(nbrs)
    for node in all_nodes:
        adj.setdefault(node, [])

    return adj


# Kreiranje MultiGraph-a na osnovu liste susedstva, ali uz ocuvanje paralelnih grana.
def build_multigraph(adj: dict[int, list[int]]) -> nx.MultiGraph:
    G = nx.MultiGraph()
    G.add_nodes_from(adj.keys())

    for u in adj:
        cnt = Counter(adj[u])
        for w, mult in cnt.items(): # U MultiGraph strukturi paralelne grane se modeluju kao zasebne ivice izmedju istih cvorova.
            if u <= w: # Uslov u <= w se koristi da se ista neusmerena grana ne doda dva puta.
                for _ in range(mult):
                    G.add_edge(u, w)
    return G


adj = parse_adj(adj_text)
G = build_multigraph(adj)


# Skup suseda bez ponavljanja.
def neigh_set(adj: dict[int, list[int]], u: int) -> set[int]:
    return set(adj[u])


# Izracunavanje najmanje udaljenosti izmedju cvorova, uz pomoc BFS algoritma.
INF = 10**9
def bfs_dist(adj: dict[int, list[int]], start: int) -> dict[int, int]:
    dist = {u: INF for u in adj}
    dist[start] = 0
    q = deque([start])

    while q:
        u = q.popleft()
        for w in neigh_set(adj, u): # BFS koristi neigh_set bez ponavljanja, jer paralelne grane ne smanjuju broj koraka.
            if dist[w] == INF:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


# Izracunavanje broja razlicitih najkracih puteva između v i t.
def shortest_paths_count_multigraph(adj: dict[int, list[int]], s: int, target: int) -> tuple[int, int]:
    dist = {u: INF for u in adj}
    ways = {u: 0 for u in adj}

    dist[s] = 0
    ways[s] = 1
    q = deque([s])

    while q:
        u = q.popleft()
        cu = Counter(adj[u]) #  Multiplicitet grana iz u ka w.

        for w, mult in cu.items():
            if dist[w] == INF:
                dist[w] = dist[u] + 1
                ways[w] = ways[u] * mult
                q.append(w)
            elif dist[w] == dist[u] + 1:
                ways[w] += ways[u] * mult

    return dist[target], ways[target]


# Broj grana u indukovanom podgrafu formiranom od cvorova v, t i njihovih suseda.
def induced_edges_count(G: nx.MultiGraph, S: set[int]) -> int:
    return G.subgraph(S).number_of_edges() # subgraph(S) zadrzava i paralelne grane, pa number_of_edges() broji sve ivice sa multiplicitetom.


# Broj komponenti nakon uklanjanja cvorova.
def components_after_removal(adj: dict[int, list[int]], removed: set[int]) -> int:
    nodes = [u for u in adj if u not in removed]
    visited = set()
    comps = 0

    for start in nodes:
        if start in visited:
            continue
        comps += 1
        q = deque([start])
        visited.add(start)

        while q:
            u = q.popleft()
            for w in neigh_set(adj, u):
                if w in removed or w in visited:
                    continue
                visited.add(w)
                q.append(w)

    return comps


# Broj setnji duzine k: (A^k)[v,t], A broji paralelne grane.
def walks_length_k(G: nx.MultiGraph, v: int, t: int, k: int) -> int:
    W = nx.Graph()
    W.add_nodes_from(G.nodes())

    for a, b in G.edges():
        if W.has_edge(a, b):
            W[a][b]["weight"] += 1
        else:
            W.add_edge(a, b, weight=1)

    nodes = sorted(W.nodes())
    idx = {node: i for i, node in enumerate(nodes)}
    A = nx.to_numpy_array(W, nodelist=nodes, weight="weight", dtype=np.int64)

    Ak = np.linalg.matrix_power(A, k)
    return int(Ak[idx[v], idx[t]])


# Resavanje zadataka.
deg = dict(G.degree()) # Stepen u MultiGraph-u broji i paralelne grane

# 1 zadatak: Zbir stepena svih cvorova osim v i t.
ans1 = sum(d for u, d in deg.items() if u != v and u != t)

# 2 zadatak: Broj razlicitih najkracih puteva od v do t.
dist_vt, ans2 = shortest_paths_count_multigraph(adj, v, t)

# 3 zadatak: Simetricna razlika skupova suseda v i t.
Nv, Nt = neigh_set(adj, v), neigh_set(adj, t)
symdiff = sorted(Nv ^ Nt)
ans3 = ",".join(map(str, symdiff)) if symdiff else "/"

# 4 zadatak: Susedi v ili t sa stepenom vecim od prosecnog stepena u grafu.
avg_deg = sum(deg.values()) / G.number_of_nodes()
cand4 = sorted([x for x in (Nv | Nt) if deg[x] > avg_deg])
ans4 = ",".join(map(str, cand4)) if cand4 else "/"

# 5 zadatak: Broj grana indukovanog podgrafa nad {v,t} i svim njihovim susedima.
S = set([v, t]) | Nv | Nt
ans5 = induced_edges_count(G, S)

# 6 zadatak: Cvorovi koji su na udaljenosti najvise 3 i od v i od t.
dv, dt = bfs_dist(adj, v), bfs_dist(adj, t)
cand6 = sorted([u for u in adj if dv[u] <= 3 and dt[u] <= 3])
ans6 = ",".join(map(str, cand6)) if cand6 else "/"

# 7 zadatak: Zbir ekscentriciteta cvorova v i t.
def ecc(dist_map: dict[int, int]) -> int: # Ekscentricitet cvora definise se kao maksimalno rastojanje, od tog cvora do bilo kog drugog dostiznog cvora u grafu.
    reachable = [d for d in dist_map.values() if d < INF]
    return max(reachable) if reachable else INF

ans7 = ecc(dv) + ecc(dt)

# 8 zadatak: Broj komponenti nakon uklanjanja v i t, i jos dva cvora sa najvecim stepenom.
others = [u for u in G.nodes() if u not in (v, t)]
others.sort(key=lambda u: (-deg[u], u))
extra = others[:2]
ans8 = components_after_removal(adj, set([v, t] + extra))

# 9 zadatak : Broj puteva duzine 3 izmedju v i t.
ans9 = walks_length_k(G, v, t, 3) # Element (i,j) matrice A^k daje broj setnji duzine k izmedju cvorova i i j.

# 10 zadatak: Broj puteva duzine 10 izmedju v i t.
ans10 = walks_length_k(G, v, t, 10)


# Ispis rezultata.
print(f"v={v}, t={t}")
print("1)", ans1)
print("2)", ans2)
print("3)", ans3)
print("4)", ans4)
print("5)", ans5)
print("6)", ans6)
print("7)", ans7)
print("8)", ans8, " Uklonjeni:", ",".join(map(str, [v, t] + extra)))
print("9)", ans9)
print("10)", ans10)