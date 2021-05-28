
class Mention(object):
    '''brat relation'''
    def __init__(self, id_str, type_str, start_offset, end_offset, mention):
        self._id = id_str
        self._type = type_str
        self._start_offset = start_offset
        self._end_offset = end_offset
        self._mention = mention

    def __repr__(self):
        print_str = '#mention: { id:' + str(self._id)
        print_str += ', type:' + self._type
        print_str += ', mention:' + self._mention
        print_str += ', start:' + str(self._start_offset)
        print_str += ', end:' + str(self._end_offset)
        print_str += '}'
        return print_str

    def copy(self):
        return Mention(self._id, self._type, self._start_offset, self._end_offset, self._mention)
