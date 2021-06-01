from method_ir.grounding.path_match_score12.canoical_codes_tree import Tree, construct_canonical_codes_recursive
from common_structs.graph import Digragh
from collections import defaultdict

# _root_id = '?uri' #lcquad gold
# _root_id = '?a' #lcquad candidate query
# _root_id = '?x' #cwq gold sparql
# _root_id = '?a' #cwq candidate query


class Graph_to_Tree():

    def __init__(self, digraph, current_tree):
        self._marked = defaultdict(bool)
        self._digraph = digraph
        self._reverse_graph = digraph.reverse()
        self.dfs(current_tree=current_tree)

    def dfs(self, current_tree):
        self._marked[current_tree.nid] = True
        for v in self._digraph.get_adjacent_vertices(current_tree.nid):
            if not self._marked[v]:
                son_tree = Tree(parent=current_tree.nid, nid=v, data=v)
                edge_label = self._digraph._startend_to_edgelabel[current_tree.nid+'\t'+v]
                current_tree.add_child(son_tree, edge_label=edge_label, direction='f-s')
                self.dfs(current_tree=son_tree)

        for v in self._reverse_graph.get_adjacent_vertices(current_tree.nid):
            if not self._marked[v]:
                son_tree = Tree(parent=current_tree.nid, nid=v, data=v)
                edge_label = self._reverse_graph._startend_to_edgelabel[current_tree.nid+'\t'+v]
                current_tree.add_child(son_tree, edge_label=edge_label, direction='s-f')
                self.dfs(current_tree=son_tree)


def convert_triple_to_paths(triples, _root_id='?a'):
    digraph = Digragh()
    for triple in triples:
        digraph.add_edge(triple['subject'], triple['object'], edgelabel=triple['predicate'])
    current_tree = Tree(parent=-1, nid=_root_id, data=_root_id)
    Graph_to_Tree(digraph=digraph, current_tree=current_tree)
    root_canonical_codes = construct_canonical_codes_recursive(current_tree=current_tree, super_nid=-1, current_nid=current_tree.nid)
    return root_canonical_codes.canonical_code
