

class AssociateionNode(object):

    def __init__(self, nid):
        self.relation = ''
        self.canonical_code = []
        self.sons = []
        self.minimalsuccessorNodeId = nid
        self.father = None


class Tree(object):

    def __init__(self, parent=-1, nid=0, data='', ):
        self.parent = parent
        self.nid = nid
        self.data = data
        self.num_children = 0
        self.children = list()
        self.canonical_code = ''

    def add_child(self, child, edge_label='', direction='f-s'):
        '''
        :param child:
        :param edge_label:
        :param direction: f-s means father -> son.   s-f means son -> father
        :return:
        '''
        child.parent = self
        self.num_children += 1
        self.children.append((edge_label, direction, child))

    def get_edge(self, super_nid, current_nid):
        result_edge_label = None
        result_direction = None
        for i in range(self.num_children):
            edge_label, direction, child = self.children[i]
            if self.nid == super_nid and current_nid == child.nid:
                result_edge_label = edge_label
                result_direction = direction
            else:
                temp_edge_label, temp_result_direction = child.get_edge(super_nid=super_nid, current_nid=current_nid)
                if temp_edge_label is not None and temp_result_direction is not None:
                    result_edge_label = temp_edge_label
                    result_direction = temp_result_direction
        return result_edge_label, result_direction

    def degree_of(self, current_nid):
        degree = 0
        for i in range(self.num_children):
            edge_label, direction, child = self.children[i]
            if current_nid == child.nid:
                degree = child.num_children
                break
            else:
                temp_degree = child.degree_of(current_nid=current_nid)
                if temp_degree != 0:
                    degree = temp_degree
        return degree

    def get_children(self, current_nid):
        children = None
        if current_nid == self.nid:
            return self.children
        for _, _, child in self.children:
            if child.nid == current_nid:
                return child.children
            else:
                temp_children = child.get_children(current_nid=current_nid)
                if temp_children is not None:
                    children = temp_children
        return children


# min_value = -sys.maxsize - 1
SupperRootID = -1
def construct_canonical_codes_recursive(current_tree, super_nid, current_nid):
    node = AssociateionNode(current_nid)

    if super_nid != SupperRootID:  #is not root
        edge_label, edge_direction = current_tree.get_edge(super_nid=super_nid, current_nid=current_nid)
        if edge_direction == 'f-s':
            node.relation = edge_label
        else:
            node.relation = '-'+edge_label

    if current_tree.degree_of(current_nid)==0 and super_nid != SupperRootID: #leaf node
        node.canonical_code.append(current_nid)

    else: #no leaf node
        for edge_label, direction, child in current_tree.get_children(current_nid):
            current_nid_neighbor_nid = child.nid
            if current_nid_neighbor_nid != super_nid:
                node.sons.append(construct_canonical_codes_recursive(
                    current_tree=current_tree, super_nid=current_nid, current_nid=current_nid_neighbor_nid))

        node.canonical_code.append(current_nid)
        node.sons.sort(key=lambda x: x.minimalsuccessorNodeId, reverse=False)  # False利用lambda表达式对单个属性进行排序
        for son in node.sons:
            node.canonical_code.append(son.relation)
            node.canonical_code.extend(son.canonical_code)
        node.minimalsuccessorNodeId = node.sons[0].minimalsuccessorNodeId

    return node


if __name__ == '__main__':
    '''
    0,acceptedAt
    1,attended
    2,firstAuthor
    3,isA
    4,isAuthorOf
    5,knows
    6,memberOfPC
    7,Alice
    8,Bob
    9,Chris
    10,DESWeb
    11,Dan
    12,Ellen
    13,Frank
    14,ICDE
    15,PaperA
    16,PaperB
    17,Conference
    18,Meeting
    19,Paper
    20,Person
    21,Student
    22,Workshop
    '''

    root_tree = Tree(parent=-1, nid=7, data='Alice')
    _16_tree = Tree(parent=7, nid=16, data='PaperB')
    root_tree.add_child(_16_tree, edge_label='firstAuthor', direction='s-f')

    _10_tree = Tree(parent=16, nid=10, data='DESWeb')
    _16_tree.add_child(_10_tree, edge_label='acceptedAt', direction='f-s')

    _8_tree = Tree(parent=10, nid=8, data='Bob')
    _9_tree = Tree(parent=10, nid=9, data='Chris')
    _10_tree.add_child(_8_tree, edge_label='memberOfPC', direction='f-s')
    _10_tree.add_child(_9_tree, edge_label='attended', direction='s-f')
    root_canonical_codes = construct_canonical_codes_recursive(current_tree=root_tree, super_nid=SupperRootID, current_nid=7)
    print(root_canonical_codes.canonical_code)

