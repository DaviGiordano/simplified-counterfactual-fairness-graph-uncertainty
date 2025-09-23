import itertools as itr
import unittest
from collections import defaultdict

import networkx as nx
import pydot


def all_dags_from_cpdag(
    cpdag_dot: pydot.Dot, all_nodes: list[str] = []
) -> list[nx.DiGraph]:
    """List every DAG in the MEC of the given CPDAG.

    Args:
        cpdag_dot: A pydot.Dot representation of the CPDAG
        all_nodes: Optional list of additional nodes to include in the result,
                   even if they don't appear in the CPDAG

    Returns:
        A list of all possible DAGs in the Markov equivalence class
    """

    # --- helper ------------------------------------------------------------
    def is_undirected(e):
        a = e.get_attributes()
        return a.get("dir") == "none" or (
            a.get("arrowhead") == "none" and a.get("arrowtail") == "none"
        )

    def v_structures(arcs, skel):
        pa = defaultdict(set)
        for u, v in arcs:
            pa[v].add(u)
        vs = set()
        for c, ps in pa.items():
            for p1, p2 in itr.combinations(ps, 2):
                if frozenset((p1, p2)) not in skel:
                    vs.add((p1, c, p2))
        return vs

    # --- parse CPDAG -------------------------------------------------------
    arcs, undirected_pairs, nodes = set(), set(), set()
    for e in cpdag_dot.get_edges():
        s, t = e.get_source().strip('"'), e.get_destination().strip('"')
        nodes.update((s, t))
        if is_undirected(e):
            undirected_pairs.add(frozenset((s, t)))
        else:
            arcs.add((s, t))

    # Add any additional nodes from all_nodes parameter
    if all_nodes != []:
        nodes.update(all_nodes)

    undirected = [tuple(p) for p in undirected_pairs]
    skeleton = {frozenset(a) for a in arcs} | undirected_pairs
    v_cpdag = v_structures(arcs, skeleton)

    # --- DFS over orientations --------------------------------------------
    dag_sets, dags = set(), []

    def dfs(i, cur):
        if i == len(undirected):  # all edges oriented
            g = nx.DiGraph()
            g.add_nodes_from(nodes)
            g.add_edges_from(cur)
            if (
                nx.is_directed_acyclic_graph(g)
                and v_structures(cur, skeleton) == v_cpdag
            ):
                fs = frozenset(cur)
                if fs not in dag_sets:
                    dag_sets.add(fs)
                    dags.append(g)
            return
        u, v = undirected[i]
        dfs(i + 1, cur | {(u, v)})
        dfs(i + 1, cur | {(v, u)})

    dfs(0, arcs.copy())
    return dags


# -------------------------------------------------------------------------
#                               Unit tests
# -------------------------------------------------------------------------
def _v(dag):
    """v-structures helper for tests"""
    out = set()
    for c in dag.nodes:
        ps = list(dag.predecessors(c))
        for p1, p2 in itr.combinations(ps, 2):
            if not dag.has_edge(p1, p2) and not dag.has_edge(p2, p1):
                out.add((p1, c, p2))
    return out


class TestCPDAG(unittest.TestCase):
    # simple factories ------------------------------------------------------
    @staticmethod
    def undirected_edge_graph(edge_list):
        g = pydot.Dot()
        # for n in nodes:
        #     g.add_node(pydot.Node(n))
        for u, v in edge_list:
            g.add_edge(pydot.Edge(u, v, arrowhead="none", arrowtail="none"))
        return g

    @staticmethod
    def triangle():
        return TestCPDAG.undirected_edge_graph([("A", "B"), ("A", "C"), ("B", "C")])

    @staticmethod
    def undirected_chain():
        return TestCPDAG.undirected_edge_graph([("X1", "X2"), ("X1", "X3")])

    @staticmethod
    def single_edge():
        return TestCPDAG.undirected_edge_graph([("X", "Y")])

    @staticmethod
    def chain():
        g = pydot.Dot(graph_type="digraph")
        for n in "ABC":
            g.add_node(pydot.Node(n))
        g.add_edge(pydot.Edge("A", "B"))
        g.add_edge(pydot.Edge("B", "C"))
        return g

    @staticmethod
    def user_cpdag():
        nodes = ("A", "X1", "X2", "Y")
        edges = [("A", "X1"), ("X1", "X2"), ("X1", "Y"), ("X2", "Y")]
        return TestCPDAG.undirected_edge_graph(edges)

    # actual tests ----------------------------------------------------------
    def test_triangle(self):
        dags = all_dags_from_cpdag(self.triangle())
        self.assertEqual(len(dags), 6)
        for d in dags:
            self.assertTrue(nx.is_directed_acyclic_graph(d))
            self.assertEqual(len(_v(d)), 0)

    def test_undirected_chain(self):
        dags = all_dags_from_cpdag(self.undirected_chain())
        self.assertEqual(len(dags), 3)  # Note that v-structures are not allowed

    def test_single_edge(self):
        dags = all_dags_from_cpdag(self.single_edge())
        self.assertEqual(len(dags), 2)

    def test_chain(self):
        dags = all_dags_from_cpdag(self.chain())
        self.assertEqual(len(dags), 1)

    def test_user_cpdag(self):
        dags = all_dags_from_cpdag(self.user_cpdag())
        self.assertEqual(len(dags), 8)  # computed exhaustively
        for d in dags:
            self.assertTrue(nx.is_directed_acyclic_graph(d))
            self.assertEqual(len(_v(d)), 0)


if __name__ == "__main__":
    unittest.main()
