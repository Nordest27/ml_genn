"""Does the agent take a longer path when the short one is a trap? Strategy metrics for Snake from board states.

At each decision (snake body head-first, apple, heading, action taken):
- a move is SAFE if after it the new head can still reach the tail cell (the cell the snake vacates next; it stays
  put when the move eats the apple), with the body as walls: the snake can then follow its own tail forever, so it has
  not trapped itself. A move into a wall or the body is lethal (not safe).
- a move is GREEDY if it shortens the shortest free path (breadth-first, body as walls except the tail) to the apple.
- a DETOUR IS NEEDED when every greedy move is unsafe and at least one safe move exists.

Metrics over a run:
- detour rate: in detour-needed states, the fraction of decisions that took a safe move (a longer path that keeps an
  escape route) instead of walking greedily into the trap;
- trap rate: the fraction of all decisions where the agent chose an unsafe, non-lethal move although a safe move
  existed (it walked into a pocket);
- avoidable deaths: the fraction of deaths where a safe move existed at the last decision.
References: a greedy policy has detour rate 0 by construction; a random non-lethal policy gives the chance level.

    from snake_strategy import evaluate
    stats = evaluate(bodies, apples, headings, actions, board, dones)

Directions: 0 left, 1 up, 2 right, 3 down (snake.py's absolute actions); a reversal is ignored (the snake goes on).
"""
from collections import deque

import numpy as np

DIRS = [(0, -1), (-1, 0), (0, 1), (1, 0)]          # left, up, right, down


def effective(action, heading):
    """snake.py ignores a move straight back into the neck."""
    return heading if (action + 2) % 4 == heading else action


def bfs(board, start, goal, walls):
    """Shortest path length from start to goal on a board x board grid avoiding walls (None if unreachable)."""
    if start == goal:
        return 0
    seen, q = {start}, deque([(start, 0)])
    while q:
        (y, x), d = q.popleft()
        for dy, dx in DIRS:
            n = (y + dy, x + dx)
            if not (0 <= n[0] < board and 0 <= n[1] < board) or n in seen or n in walls:
                continue
            if n == goal:
                return d + 1
            seen.add(n); q.append((n, d + 1))
    return None


def move_outcome(body, apple, move, board):
    """(lethal, safe, eats) for moving the head one step in direction move."""
    hy, hx = body[0]
    new = (hy + DIRS[move][0], hx + DIRS[move][1])
    eats = new == apple
    if not (0 <= new[0] < board and 0 <= new[1] < board) or new in body[:-1]:
        return True, False, eats
    new_body = [new] + (list(body) if eats else list(body[:-1]))
    tail = new_body[-1]
    if len(new_body) == 1:
        return False, True, eats
    safe = bfs(board, new, tail, set(new_body[1:-1])) is not None
    return False, safe, eats


def classify(body, apple, heading, action, board):
    """Per decision: whether a detour was needed, and whether the move taken was safe / greedy / lethal."""
    body = [tuple(c) for c in body]
    walls = set(body[1:-1])
    d0 = bfs(board, body[0], tuple(apple), walls) if apple is not None else None
    moves = {}
    for m in range(4):
        if (m + 2) % 4 == heading and len(body) > 1:
            continue                                     # reversal: not a real option
        lethal, safe, _ = move_outcome(body, tuple(apple) if apple is not None else None, m, board)
        greedy = False
        if not lethal and d0 is not None:
            hy, hx = body[0]
            new = (hy + DIRS[m][0], hx + DIRS[m][1])
            d1 = bfs(board, new, tuple(apple), set(body[:-1]) - {new}) if new != tuple(apple) else 0
            greedy = d1 is not None and d1 < d0
        moves[m] = {"lethal": lethal, "safe": safe, "greedy": greedy}
    taken = moves.get(effective(action, heading), moves.get(heading))
    greedy_moves = [v for v in moves.values() if v["greedy"]]
    any_safe = any(v["safe"] for v in moves.values())
    detour_needed = bool(greedy_moves) and not any(v["safe"] for v in greedy_moves) and any_safe
    return {"detour_needed": detour_needed, "any_safe": any_safe, "taken_safe": taken["safe"],
            "taken_greedy": taken["greedy"], "taken_lethal": taken["lethal"]}


def evaluate(bodies, apples, headings, actions, board, dones=None):
    """Aggregate strategy metrics over consecutive decisions (dones[i]: decision i ended the episode)."""
    n_detour = n_detour_ok = n_trap = n_dec = n_death = n_avoid = 0
    for i, (body, apple, heading, action) in enumerate(zip(bodies, apples, headings, actions)):
        if len(body) == 0:
            continue
        c = classify(body, apple, int(heading), int(action), board)
        n_dec += 1
        if c["detour_needed"]:
            n_detour += 1
            n_detour_ok += c["taken_safe"]
        if c["any_safe"] and not c["taken_safe"] and not c["taken_lethal"]:
            n_trap += 1
        if dones is not None and dones[i]:
            n_death += 1
            n_avoid += c["any_safe"]
    return {"decisions": n_dec, "detour_needed": n_detour,
            "detour_rate": n_detour_ok / n_detour if n_detour else float("nan"),
            "trap_rate": n_trap / n_dec if n_dec else float("nan"),
            "deaths": n_death, "avoidable_deaths": n_avoid / n_death if n_death else float("nan")}


def reference_policies(bodies, apples, headings, board, seed=0):
    """Detour rate of a greedy policy (first greedy move, else any non-lethal) and a random non-lethal policy on the
    same states: what the agent's detour rate should be compared with."""
    rng = np.random.default_rng(seed)
    out = {}
    for name in ("greedy", "random_nonlethal"):
        ok = n = 0
        for body, apple, heading in zip(bodies, apples, headings):
            if len(body) == 0:
                continue
            body = [tuple(c) for c in body]
            opts = {}
            for m in range(4):
                if (m + 2) % 4 == int(heading) and len(body) > 1:
                    continue
                opts[m] = classify(body, apple, int(heading), m, board)
            if not opts or not next(iter(opts.values()))["detour_needed"]:
                continue
            nonlethal = [m for m, c in opts.items() if not c["taken_lethal"]] or list(opts)
            if name == "greedy":
                g = [m for m, c in opts.items() if c["taken_greedy"]]
                m = g[0] if g else nonlethal[0]
            else:
                m = int(rng.choice(nonlethal))
            n += 1
            ok += opts[m]["taken_safe"]
        out[name] = ok / n if n else float("nan")
    return out


SAFETY_NAMES = ["safe_left", "safe_up", "safe_right", "safe_down", "detour_needed"]


def safety_labels(body, apple, heading, board):
    """Decoding targets for one state: whether moving in each absolute direction is safe (the tail stays reachable;
    lethal moves and the reversal into the neck count as unsafe), and whether a detour is needed. body: head-first
    cells, rows of -1 ignored (padding); apple: (y, x) or (-1, -1)."""
    body = [tuple(int(v) for v in c) for c in np.asarray(body) if c[0] >= 0]
    apple = tuple(int(v) for v in apple) if apple is not None and apple[0] >= 0 else None
    if not body:
        return np.full(5, -1, np.int8)
    safe = [int(not move_outcome(body, apple, m, board)[0] and move_outcome(body, apple, m, board)[1]) for m in range(4)]
    detour = int(classify(body, apple, int(heading), int(heading), board)["detour_needed"]) if apple else 0
    return np.array(safe + [detour], np.int8)


def flood_area(body, start, board):
    """Free cells reachable from start (the body except its last cell counts as wall)."""
    walls = set(body[:-1])
    if start in walls or not (0 <= start[0] < board and 0 <= start[1] < board):
        return 0
    seen, q = {start}, deque([start])
    while q:
        y, x = q.popleft()
        for dy, dx in DIRS:
            n = (y + dy, x + dx)
            if 0 <= n[0] < board and 0 <= n[1] < board and n not in seen and n not in walls:
                seen.add(n); q.append(n)
    return len(seen)


def state_variables(body, apple, heading, board):
    """Battery of decoding targets for one state (body head-first, -1 padding ignored). Returns (classes, values):
    classes: per direction 0 lethal / 1 trap (non-lethal, tail unreachable) / 2 safe; values: free area reachable
    after each move (fraction of the board), path distance to the apple around the body, snake length, tail dy/dx."""
    body = [tuple(int(v) for v in c) for c in np.asarray(body) if c[0] >= 0]
    apple = tuple(int(v) for v in apple) if apple is not None and apple[0] >= 0 else None
    cls, area = [], []
    hy, hx = body[0]
    for m in range(4):
        lethal, safe, _ = move_outcome(body, apple, m, board)
        cls.append(0 if lethal else (2 if safe else 1))
        area.append(0.0 if lethal else flood_area(body, (hy + DIRS[m][0], hx + DIRS[m][1]), board) / board ** 2)
    dist = bfs(board, body[0], apple, set(body[1:-1])) if apple else None
    ty, tx = body[-1]
    values = area + [float(dist) if dist is not None else float(2 * board), float(len(body)), float(ty - hy),
                     float(tx - hx)]
    return np.array(cls, np.int8), np.array(values, np.float32)


STATE_CLASS_NAMES = ["move_left", "move_up", "move_right", "move_down"]       # lethal / trap / safe
STATE_VALUE_NAMES = ["area_left", "area_up", "area_right", "area_down", "path_to_apple", "length", "tail_dy",
                     "tail_dx"]
