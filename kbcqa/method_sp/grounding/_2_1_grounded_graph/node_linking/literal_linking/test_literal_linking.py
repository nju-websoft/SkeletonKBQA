from common.hand_files import read_structure_file


def test_graphq(structure_list):
    literal_node_set = set()
    for i, structure in enumerate(structure_list):
        for ungrounded_graph in structure.ungrounded_graph_forest:
            for _2_1_grounded_graph in ungrounded_graph.get_grounded_graph_forest():
                for _2_1_grounded_graph_node in _2_1_grounded_graph.nodes:
                    if _2_1_grounded_graph_node.node_type == 'literal':
                        literal_node_set.add(_2_1_grounded_graph_node.id)
                        # print(_2_1_grounded_graph_node.id, _2_1_grounded_graph_node.type_class)
    filter_list = ['present', 'current', '2019 INTERSECT P1M^^http://www.w3.org/2001/XMLSchema#datetime',
                   '4wd', 'march the 31th , 2005', 'currently', '1768.0 kg', '129.54 mm', 'april the 7th , 1995',
                   '4686.3 mm', '1905-born', '40d', 'april the 26th , 1882', '243.01 km',
                   '2019-FA^^http://www.w3.org/2001/XMLSchema#datetime', '243.01 km/h',
                   '2019^^http://www.w3.org/2001/XMLSchema#datetime', '1100.0 kg', 'the past',
                   '1912-SU^^http://www.w3.org/2001/XMLSchema#datetime', '128 bits', '4000 customers',
                   '2008-SU^^http://www.w3.org/2001/XMLSchema#datetime', '82.0 kg', 'sept. the 20th , 2008',
                   '1843-FA^^http://www.w3.org/2001/XMLSchema#datetime']
    return literal_node_set


if __name__ == '__main__':
    # output_file_folder = 'D:/dataset/dataset_graphquestions/' \
    #                      'output_graphq_e2e/output_graphq_sp_skeleton_multi_strategy_e2e_sp1/'
    # structure_with_2_1_grounded_graph_file = output_file_folder + 'structures_with_2_1_grounded_graph_test_0124.json'
    # structure_list = read_structure_file(structure_with_2_1_grounded_graph_file)
    # test_literal_node_set = test_graphq(structure_list=structure_list)
    # structure_with_2_1_grounded_graph_file = output_file_folder + 'structures_with_2_1_grounded_graph_train_0124.json'
    # structure_list = read_structure_file(structure_with_2_1_grounded_graph_file)
    # train_literal_node_set = test_graphq(structure_list=structure_list)
    # all_literal_node_set = test_literal_node_set.union(train_literal_node_set)
    # for literal_id in all_literal_node_set:
    #     print(literal_id)

    output_file_folder = 'D:/dataset/dataset_cwq_1_1/' \
                         'output_cwq_e2e/output_cwq_sp_skeleton_multi_strategy_e2e_sp1/'
    structure_with_2_1_grounded_graph_file = output_file_folder + 'structures_with_2_1_grounded_graph_test_skeleton_system_el_0127.json'
    structure_list = read_structure_file(structure_with_2_1_grounded_graph_file)
    test_literal_node_set = test_graphq(structure_list=structure_list)
    for literal_id in test_literal_node_set:
        print(literal_id)

