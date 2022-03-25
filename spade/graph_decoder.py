import torch
import numpy as np
from queue import Queue


def dfs_queue(adj, visited=None, starts=None):
    # Preparation
    (
        n,
        _,
    ) = adj.shape
    queue = Queue()
    if visited is None:
        visited = [False for _ in range(n)]

    def recurse(adj, start, visited, queue):
        visited[start] = True
        neighbors = np.where(adj[start, :])[0]
        if len(neighbors) == 0:
            return
        for j in np.where(adj[start, :])[0]:
            if not visited[j]:
                queue.put((start, j))
                recurse(adj, j, visited, queue)

    # Allow chosing where to start
    if starts is None:
        starts = range(n)

    for start in starts:
        if visited[start]:
            continue
        recurse(adj, start, visited, queue)
    return queue


def parse_graph(score, texts, fields, strict=True):
    rel_s, rel_g = score[0], score[1]
    ntexts = len(texts)
    nfields = len(fields)

    #  Label to text and Text 2 text
    rel_s_l2t, rel_s_t2t = rel_s[0:nfields, :], rel_s[nfields:, :]
    rel_g_l2t, rel_g_t2t = rel_s[0:nfields, :], rel_g[nfields:, :]

    # Link node type to representing nodes
    node_types = {}
    for (i, j) in zip(*np.where(rel_s_l2t)):
        node_types[j] = i

    # Link label to the nodes following the representing nodes
    starts = node_types.keys()
    queue = dfs_queue(rel_s_t2t, starts=starts)
    queue_backup = Queue()  # This is needed later
    while not queue.empty():
        (i, j) = queue.get()
        queue_backup.put((i, j))
        # print(texts[i], texts[j])
        # continue
        if strict:
            assert i in node_types, (
                f"can't find label for {texts[i]} --> {texts[j]}, the labels:\n"
                + pformat([(texts[k], fields[v]) for (k, v) in node_types.items()])
            )
        node_types[j] = node_types[i]

    # Link nodes in side a group
    groups = []
    queue = dfs_queue(rel_g_t2t, starts=starts)
    while not queue.empty():
        (i, j) = queue.get()
        # Find the belonging group
        the_group = [group for group in groups if i in group]
        if strict:
            assert (
                len(the_group) < 2
            ), f"{texts[i]} can be found in both {pformat([[texts[i] for i in g] for g in the_group])}"
        if len(the_group) == 0:
            groups.append([i, j])
        else:
            the_group[0].append(j)

    # Link rel_s inside group
    while not queue_backup.empty():
        (i, j) = queue_backup.get()
        the_group = [group for group in groups if i in group]
        if strict:
            assert (
                len(the_group) < 2
            ), f"{texts[i]} can be found in both {pformat([[texts[i] for i in g] for g in the_group])}"
        if len(the_group) == 0:
            groups.append([i, j])
        elif j not in the_group[0]:
            the_group[0].append(j)

    # Standalone nodes goes in a standalone group
    for (inode, ifield) in node_types.items():
        the_group = [group for group in groups if inode in group]
        if strict:
            assert (
                len(the_group) < 2
            ), f"{texts[inode]} can be found in both {pformat([[texts[i] for i in g] for g in the_group])}"
        if len(the_group) == 0:
            groups.append([inode])

    # Auxilary prettified results
    aux_results = []
    results = []

    def aux(i):
        if i > len(node_types) - 1:
            ntype = "Unknown"
        elif node_types[i] > len(fields) - 1:
            ntype = "Unknown"
        else:
            ntype = fields[node_types[i]]

        return ntype

    # return aux_results

    for group in groups:
        try:
            aux_results.append([(texts[i], aux(i)) for i in group])
        except Exception:
            pass

    # Prettified results
    for result in aux_results:
        parsed = {}
        # Join text nodes with same field values
        for (text, field) in result:
            parsed[field] = parsed.get(field, [])
            parsed[field].append(text)
        for field in parsed:
            parsed[field] = " ".join(parsed[field])
        results.append(parsed)
    return results


# for (i, _) in enumerate(data):
#     try:
#         p = parse_graph(torch.tensor(
#             data[i]['label']), data[i]['text'], data[i]['fields'])
#         if i == 84:
#             print(f"{i+1}. {data[i]['data_id']}")
#             pprint(p)
#     except Exception as e:
#         import traceback
#         print(i, data[i]['data_id'])
#         traceback.print_exc()
