
class Relation(object):
    '''brat relation'''
    def __init__(self, relation_id_str, relation_type, relation_start_index, relation_end_index):
        self._relation_id_str = relation_id_str
        self._relation_type = relation_type
        self._relation_start_index = relation_start_index
        self._relation_end_index = relation_end_index

    def __repr__(self):
        print_str = '#relation:{ id:' + self._relation_id_str
        print_str += ', type:' + self._relation_type
        print_str += ', start:' + self._relation_start_index
        print_str += ', end:' + self._relation_end_index
        print_str += '}'
        return print_str

    def copy(self):
        return Relation(self._relation_id_str, self._relation_type, self._relation_start_index, self._relation_end_index)
