

class Token:
    '''a token: index, value, layer, isEnd'''

    def __init__(self, index, token, isEnd=False):
        # self.left = None self.right = None self.father = None self.children = None
        self.index = index
        self.value = token
        self.shield = False  #same as the delete token
        self.layer = -1
        self.ner_tag = None  #entity_I, entity_E, class_I, class_E, literal_I, literal_E
        self.isEnd = isEnd

    def __str__(self):
        return '{}\t{}\t{}'.format(self.index, self.value, self.ner_tag)


class Span:
    '''A single tree node: Each node has a unique key(identifier), a value(content) and a list of children'''

    def __init__(self, id, start_position=None, end_position=None, isRoot=False, tokens=None, headword_position=None, headword_relation=None):
        self.__id = id
        self.__start_position = start_position
        self.__end_position = end_position
        self.__children = []
        self.__isRoot = isRoot
        self.__headword_position = headword_position
        self.__headword_relation = headword_relation
        self.__tokens = tokens
        self.state = None  #c, h

    @property
    def start_position(self): return self.__start_position

    @property
    def end_position(self): return  self.__end_position

    @property
    def headword_position(self): return self.__headword_position

    @property
    def headword_relation(self): return self.__headword_relation

    @property
    def content(self):
        content = ''
        for token in self.__tokens:
            # if self.isRoot and not token.shield:
            #     content = content + token.value + ' '
            # elif not self.isRoot:
            content = content + token.value + ' '
        return content.strip()

    @property
    def tokens(self): return self.__tokens

    @property
    def id(self): return self.__id

    @property
    def isRoot(self): return self.__isRoot

    @property
    def children(self): return self.__children

    @property
    def isTerminal(self):
        if len(self.__children) == 0:
            return True
        else:
            return False

    def set_tokens(self, tokens):
        self.__tokens = tokens

    def set_headword_position(self, headword_position):
        self.__headword_position = headword_position

    def set_headword_relation(self, headword_relation):
        self.__headword_relation = headword_relation

    def add_child(self, son_id):
        '''children are stored by key'''
        self.__children.append(son_id)

    def show_line(self, tree):
        '''print tree'''
        print_str = '['+str(self.content) + '['+str(self.headword_position)+'{'+str(self.headword_relation)+'}'+']'
        for child in self.children:
            print_str += tree.nodes[child].show_line(tree)
        print_str += ']'
        return print_str

    def __str__(self):
        return 'ID:'+str(self.id)\
               + '\tPosition:'+ str(self.__start_position) +'-'+ str(self.__end_position) \
               + '\tContent:'+ self.content \
               + '\tHeadword:'+str(self.headword_position)


class SpanTree(object):
    '''A collection of tree nodes and relations'''

    def __init__(self, tokens=None):
        self.__nodes = {} #id: span
        self.__root_node = None
        self.__tokens = tokens

    @property
    def tokens(self): return self.__tokens

    @property
    def nodes(self): return self.__nodes

    def add_span_node(self, id=None, head_tail_position=None, isRoot=None, tokens=None):
        if id not in self.__nodes.keys():
            self.__nodes[id] = Span(id=id, start_position=head_tail_position[0], end_position=head_tail_position[1], tokens=tokens, isRoot=isRoot)
        if isRoot: self.root_node = id
        return self.__nodes[id]

    def add_child_rel_with_headword(self, father_id=None, son_id=None, headword_position=None, headword_relation=None):
        self.__nodes[father_id].add_child(son_id) #父亲span下追加孩子
        self.__nodes[son_id].set_headword_position(headword_position) #孩子span下加修饰词的位置信息
        self.__nodes[son_id].set_headword_relation(headword_relation) #孩子span下加修饰词的关系信息

    def get_span_by_id(self, id):
        '''use in tree-lstm'''
        current_node = None
        for id_temp, node in self.nodes.items():
            if id_temp == id:
                current_node = node
        return current_node

    def get_span_by_ids(self, ids):
        '''use in tree-lstm'''
        children_nodes = []
        for id_ in ids:
            children_nodes.append(self.get_span_by_id(id_))
        return children_nodes

    def get_root_span_node(self):
        '''look for root'''
        root_node = None
        for id, node in self.nodes.items():
            if node.isRoot:
                root_node = node
            # for child in node.children:
            #     print('\t\tson:',tree.nodes[child])
        return root_node

    def get_children_spans_by_fatherspan(self, father_span):
        children_ids = father_span.children
        children_spans = []
        for child_id in children_ids:
            children_spans.append(self.__nodes[child_id])
        return children_spans

    def get_father_span_by_sonid(self, son_id):
        father_span_node = None
        for id, node in self.nodes.items():
            for current_child_id in node.children:
                if current_child_id == son_id:
                    father_span_node = node
        return father_span_node

    def __str__(self):
        '''print span tree'''
        root_node = self.get_root_span_node()
        return root_node.show_line(self)

    def dfs_traverse(self, current_span):
        lines = []
        # lines.append(current_span.content)
        children_spans = self.get_children_spans_by_fatherspan(current_span)
        for child_span in children_spans:
            lines.extend(self.dfs_traverse(child_span))
            lines.append('%s\t%d\t%s' % (child_span.content, child_span.headword_position, child_span.headword_relation)) #LAS
            # lines.append('%s\t%d' % (child_span.content, child_span.headword_position)) #UAS
        return lines

    def dfs_traverse_update(self, current_span):
        lines = []
        printed_child_span_list = []
        children_spans = self.get_children_spans_by_fatherspan(current_span)
        for child_span in children_spans:
            temp_lines, temp_printed_child_span_list = self.dfs_traverse_update(child_span)
            lines.extend(temp_lines)
            printed_child_span_list.extend(temp_printed_child_span_list)
            content = ''
            for i in range(len(self.tokens)):
                # 孩子的其他子孙
                if child_span.start_position <= i <= child_span.end_position and not self.tokens[i] in child_span.tokens:
                    continue
                is_print = False
                for printed_child_span in printed_child_span_list:
                    if printed_child_span.start_position <= i <= printed_child_span.end_position:
                        is_print = True
                if is_print:
                    continue
                content += self.tokens[i].value + ' '
            lines.append('%s\t%s\t%d\t%s' % (content.strip(), child_span.content, child_span.headword_position, child_span.headword_relation))
            printed_child_span_list.append(child_span)
        return lines, printed_child_span_list
