from collections import defaultdict
from common_structs.grounded_graph import GrounedGraph
from common_structs.depth_first_paths import DepthFirstPaths
from method_sp.grounding._2_2_grounded_graph.grounding_online_sp import sp_online_utils


#utils
class ConstructGroundedGraph():

    '''one path question'''
    def __init__(self, graph,
                 start_vertex,
                 grounded_nodes,
                 nodenid_to_instances_dict,
                 son_father_to_paths_list,
                 mediator_num,
                 sequence_upper):
        self._graph = graph
        self._start_vertex = start_vertex
        self._grounded_nodes = grounded_nodes
        self._nodenid_to_instances_dict = nodenid_to_instances_dict
        self._son_father_to_paths_list = son_father_to_paths_list

        self._mediator_num = mediator_num
        self._sequence_upper = sequence_upper

        self._grounded_graphs = []
        self._current_grounded_graph = None

        self._grounded_sequence_completed_list = []
        self._path_sequence = []
        self._path_list = []
        self.comput_paths()
        self._nid_to_mid_dict = dict()
        print ('enumerate all sequence...')
        self.recursion_generate_grounded_graph(index=0)

    # 1
    def comput_paths(self):
        dfp = DepthFirstPaths(self._graph, self._start_vertex)
        for node_nid in self._graph.vertices():
            if node_nid == self._start_vertex:
                continue
            if self._graph.degree(node_nid) > 1:
                continue
            if dfp.has_path_to(node_nid):
                path_to_list = [i for i in dfp.path_to(node_nid)]
                path_to_list.reverse()
                self._path_list.append(path_to_list)

    # 2
    def recursion_generate_grounded_graph(self, index=0, previous_end_id=None, is_same_path=True):
        '''recursion enum all candidate equences'''
        if len(self._grounded_sequence_completed_list) > self._sequence_upper:
            return
        if index == len(self._son_father_to_paths_list):
            # filter reverse property
            has_reverse_property = sp_online_utils.has_reverse_property(self._path_sequence)
            if not has_reverse_property:
                all_sequence_size = len(self._grounded_sequence_completed_list)
                if all_sequence_size % 1000 == 0:
                    print ('#sequence_size:\t', all_sequence_size)
                self._grounded_sequence_completed_list.append(self._path_sequence.copy())
            return
        (son_nid, father_nid), paths = self._son_father_to_paths_list[index]
        for path in paths:
            spo = path.split('\t')
            start_id = spo[0]
            end_id = spo[len(spo)-1]
            if end_id in self._nodenid_to_instances_dict[father_nid] and start_id in self._nodenid_to_instances_dict[son_nid]:
            # id must be in the instances of nodenid
                if is_same_path: #is the same path
                    if previous_end_id is not None and start_id != previous_end_id:
                        continue
                else:  #is not same path
                    if self.get_path_length_by_son_nid(son_nid) == 1 and previous_end_id is not None \
                            and end_id != previous_end_id: #the another path's length is 1, if spo[2] != previous_end_id: continue
                        continue

                if father_nid in self._nid_to_mid_dict:
                    intersection=True
                    if self._nid_to_mid_dict[father_nid] != end_id:
                        continue
                else:
                    intersection=False

                self._nid_to_mid_dict[son_nid] = start_id
                self._nid_to_mid_dict[father_nid] = end_id
                self._path_sequence.append(path)
                self.recursion_generate_grounded_graph(index=index + 1, previous_end_id=end_id, is_same_path=self.is_same_path_(father_nid, index))
                if len(self._grounded_sequence_completed_list) > self._sequence_upper:
                    return
                self._path_sequence.pop()
                if intersection:
                    if son_nid in self._nid_to_mid_dict:
                        self._nid_to_mid_dict.pop(son_nid)
                else:
                    if son_nid in self._nid_to_mid_dict:
                        self._nid_to_mid_dict.pop(son_nid)
                    if father_nid in self._nid_to_mid_dict:
                        self._nid_to_mid_dict.pop(father_nid)

    # 3
    def all_edge_sequence(self):
        '''every equence to grounded graph'''
        for i, path_sequence in enumerate(self._grounded_sequence_completed_list):
            if i % 1000 == 0:
                print('#grounded_graph sequence:\t', i, path_sequence) #1 ['m.04cqj5\ttime.time_zone.locations_in_this_time_zone\tm.02j9z', 'm.0154j\tlocation.location.containedby\tm.02j9z']
            edge_to_path_dict = dict()
            mediator_count = 0
            for a in range(len(path_sequence)):
                (start_id, end_id), _ = self._son_father_to_paths_list[a]
                edge_to_path_dict[(start_id, end_id)] = path_sequence[a]
                if len(path_sequence[a].split('\t')) > 3:
                    mediator_count +=1
            if mediator_count > self._mediator_num:
                continue
            #continue reverse property, there is bug, do not consider mediator edge.
            # is_contain_reverse_property = False
            # for path_a in edge_to_path_dict.values():
            #     p1 = path_a.split('\t')[1]
            #     for path_b in edge_to_path_dict.values():
            #         p2 = path_b.split('\t')[1]
            #         if p1 == p2: continue
            #         p1_reverse = grounding_path_ywsun.get_reverse_property(p1)
            #         if p1_reverse == p2:
            #             is_contain_reverse_property = True
            # if is_contain_reverse_property: continue
            current_grounded_graph = self.edge_sequence_to_graph(_graph=self._graph, _start_vertex=self._start_vertex,
                                                                _grounded_nodes=self._grounded_nodes, edge_to_path_dict=edge_to_path_dict, id=i)
            self._grounded_graphs.append(current_grounded_graph)

    # 4
    def get_grounded_graph(self):
        return self._grounded_graphs

    # 3.5
    def edge_sequence_to_graph(self, _graph, _start_vertex, _grounded_nodes, edge_to_path_dict, id):
        '''sequence to grounded graph'''
        current_grounded_graph = GrounedGraph(nodes=list(), edges=list())
        marked = defaultdict(bool)
        sp_online_utils.dfs(_grounded_nodes, current_grounded_graph, marked, _graph, _start_vertex, edge_to_path_dict)
        current_grounded_graph.set_grounded_query_id(id)
        return current_grounded_graph

    def is_same_path_(self, father_nid, index):
        '''across path'''
        is_same_path = True
        father_node = sp_online_utils.search_one_node_in_nodes_by_nid(self._grounded_nodes, father_nid)
        if sp_online_utils.is_question_node(father_node) and index + 1 != len(self._son_father_to_paths_list):
            is_same_path = False
        return is_same_path

    def get_path_length_by_son_nid(self, son_nid):
        '''only length 1, comparative'''
        length = 1
        for path in self._path_list:
            if son_nid in path:
                length = len(path)-1
        return length

