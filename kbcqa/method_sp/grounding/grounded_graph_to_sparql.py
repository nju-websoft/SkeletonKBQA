

def grounded_graph_to_sparql(grounded_graph=None, q_function=None, q_compositionality_type='simple', q_mode='lcquad'):
    if q_mode in ['lcquad']:
        return _grounded_graph_to_sparql_dbpedia(grounded_graph, q_function=q_function, q_compositionality_type=q_compositionality_type)
    elif q_mode in ['cwq', 'graphq']:
        return _grounded_graph_to_sparql_freebase(grounded_graph, q_function=q_function, q_compositionality_type=q_compositionality_type)


def _grounded_graph_to_sparql_dbpedia(grounded_graph, q_function=None, q_compositionality_type='simple'):
    nid_nodetype_id_questionnode_function_xid = dict()
    # nodexid = 1
    sparql = 'PREFIX : <http://dbpedia.org/resource/> '
    if q_compositionality_type == 'ask':
        sparql += """ ASK WHERE { """
    id_tag = 0
    for node in grounded_graph.nodes:
        xid = "?x" + str(id_tag) #node.nid
        id_tag += 1
        if node.node_type == "class" and node.question_node == 1:
            # xid = "?x"
            if q_function == 'count':
                sparql += 'SELECT DISTINCT COUNT('+xid+') WHERE { '
            else:
                sparql += 'SELECT DISTINCT '+xid+' WHERE { '
        # else:
            # xid = "?x" + str(nodexid)
            # nodexid += 1
        nodetype_id_questionnode_function_xid = dict()
        nodetype_id_questionnode_function_xid["node_type"] = node.node_type
        nodetype_id_questionnode_function_xid["type_class"] = node.type_class
        nodetype_id_questionnode_function_xid["id"] = node.id
        nodetype_id_questionnode_function_xid["question_node"] = node.question_node
        nodetype_id_questionnode_function_xid["function"] = node.function
        nodetype_id_questionnode_function_xid["xid"] = xid
        nid_nodetype_id_questionnode_function_xid[node.nid] = nodetype_id_questionnode_function_xid

    #topic entities
    #id is grounded value, xid is variable in sparql
    for nid in nid_nodetype_id_questionnode_function_xid:
        if nid_nodetype_id_questionnode_function_xid[nid]["node_type"] == "entity":
            sparql += "VALUES " + nid_nodetype_id_questionnode_function_xid[nid]["xid"] +\
                      " { :" + nid_nodetype_id_questionnode_function_xid[nid]["id"] + " } . "
    #bgp信息
    # edge.start is nid
    for edge in grounded_graph.edges:
        sparql += (nid_nodetype_id_questionnode_function_xid[edge.start]["xid"]
                   + " <" + edge.relation + "> "
                   + nid_nodetype_id_questionnode_function_xid[edge.end]["xid"] + " .")
    #类型信息
    # for nid in nid_nodetype_id_questionnode_function_xid:
    #     if nid_nodetype_id_questionnode_function_xid[nid]["node_type"] == "class":
    #         if str(nid) in grounded_linking_dict:
    #             class_uri_dict = grounded_linking_dict[str(nid)]
    #             class_uri = None
    #             for class_uri_temp in class_uri_dict:
    #                 class_uri = class_uri_temp
    #                 break
    #             sparql += (nid_nodetype_id_questionnode_function_xid[nid]["xid"] + " <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> " + class_uri + " .")
    #聚合类处理
    # for nid in nid_nodetype_id_questionnode_function_xid:
    #     if nid_nodetype_id_questionnode_function_xid[nid]["function"] == "argmax":
    #         sparql += '} ORDER BY DESC('+nid_nodetype_id_questionnode_function_xid[nid]['xid']+') LIMIT 1'
    #         break
    #     elif nid_nodetype_id_questionnode_function_xid[nid]['function'] == 'argmin':
    #         sparql += '} ORDER BY ASC('+nid_nodetype_id_questionnode_function_xid[nid]['xid']+') LIMIT 1'
    #         break
    #     else:
    #         pass

    sparql += ("}")
    return sparql


def _grounded_graph_to_sparql_freebase(grounded_graph, q_function=None, q_compositionality_type='simple'):
    nid_nodetype_id_questionnode_function_xid = dict()
    nodexid = 1
    sparql = ''
    for node in grounded_graph.nodes:
        if node.node_type == "class" and node.question_node==1:
            xid = "?x"
            # select_sentence = """SELECT DISTINCT ?x WHERE {\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en')) \n"""
            select_sentence = """SELECT DISTINCT ?x WHERE {\nFILTER (!isLiteral(?x)) \n"""
            sparql += select_sentence
        else:
            xid = "?x"+str(nodexid)
            nodexid += 1

        nodetype_id_questionnode_function_xid = dict()
        nodetype_id_questionnode_function_xid["node_type"] = node.node_type
        nodetype_id_questionnode_function_xid["type_class"] = node.type_class
        nodetype_id_questionnode_function_xid["id"] = node.id
        nodetype_id_questionnode_function_xid["question_node"] = node.question_node
        nodetype_id_questionnode_function_xid["function"] = node.function
        nodetype_id_questionnode_function_xid["xid"] = xid
        nid_nodetype_id_questionnode_function_xid[node.nid] = nodetype_id_questionnode_function_xid

    for nid in nid_nodetype_id_questionnode_function_xid:
        if nid_nodetype_id_questionnode_function_xid[nid]["node_type"]=="class":
            pass
            # if nid_nodetype_id_questionnode_function_xid[nid]["id"] in wh_words_set:
            #     continue
            # shot down: ?x1 :type.object.type :m.06mt91
            # sparql += nid_nodetype_id_questionnode_function_xid[nid]["xid"]+" :type.object.type :"+ nid_nodetype_id_questionnode_function_xid[nid]["id"]+" .\n"
        elif nid_nodetype_id_questionnode_function_xid[nid]["node_type"]=="entity":
            sparql += "VALUES "+nid_nodetype_id_questionnode_function_xid[nid]["xid"]+" { :"+nid_nodetype_id_questionnode_function_xid[nid]["id"]+" } .\n"
    edge_sentence = ""
    pid = 0
    for edge in grounded_graph.edges:
        if edge.relation is None or edge.relation == '':
            edge_sentence += (nid_nodetype_id_questionnode_function_xid[edge.start]["xid"] + " ?p"+str(pid) +" "+ nid_nodetype_id_questionnode_function_xid[edge.end]["xid"] + " .\n")
            pid += 1
        else:
            edge_sentence += (nid_nodetype_id_questionnode_function_xid[edge.start]["xid"] + " :" + edge.relation + " " + nid_nodetype_id_questionnode_function_xid[edge.end]["xid"] + " .\n")
    sparql += edge_sentence
    sparql+=("}")
    return sparql


def grounded_graph_to_sparql_GraphQ(grounded_graph):
    sparql = ""
    nodexid = 1
    argmax_nid = ""
    argmin_nid = ""
    nid_nodetype_id_questionnode_function_xid = dict()

    for node in grounded_graph.nodes:
        if node.node_type == "class" and node.question_node==1:
            answertype=node.id
            xid= "?x0"
            select_sentence = """SELECT (?x0 AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n"""
            # print(node.function)
            if node.function=="count":
                select_sentence="""SELECT (COUNT(?x0) AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n"""
            elif node.function=="none" or node.function is None:
                select_sentence="""SELECT (?x0 AS ?value) WHERE {\nSELECT DISTINCT ?x0  WHERE { \n"""
            sparql += select_sentence
        else:
            xid = "?x"+str(nodexid)
            nodexid += 1

        nodetype_id_questionnode_function_xid = dict()
        nodetype_id_questionnode_function_xid["node_type"] = node.node_type
        nodetype_id_questionnode_function_xid["type_class"] = node.type_class
        nodetype_id_questionnode_function_xid["id"] = node.id
        nodetype_id_questionnode_function_xid["question_node"] = node.question_node
        nodetype_id_questionnode_function_xid["function"] = node.function
        nodetype_id_questionnode_function_xid["xid"] = xid
        nid_nodetype_id_questionnode_function_xid[node.nid] = nodetype_id_questionnode_function_xid

    node_sentence = ""
    for nid in nid_nodetype_id_questionnode_function_xid:
        if nid_nodetype_id_questionnode_function_xid[nid]["node_type"]=="class":
            sparql+=nid_nodetype_id_questionnode_function_xid[nid]["xid"]+":type.object.type :"+ nid_nodetype_id_questionnode_function_xid[nid]["id"]+" .\n"
            node_sentence+=nid_nodetype_id_questionnode_function_xid[nid]["xid"]+":type.object.type :"+ nid_nodetype_id_questionnode_function_xid[nid]["id"]+" .\n"

        elif nid_nodetype_id_questionnode_function_xid[nid]["node_type"]=="entity":
            sparql += "VALUES "+nid_nodetype_id_questionnode_function_xid[nid]["xid"]+" { :"+nid_nodetype_id_questionnode_function_xid[nid]["id"]+" } .\n"
            node_sentence += "VALUES "+nid_nodetype_id_questionnode_function_xid[nid]["xid"]+" { :"+nid_nodetype_id_questionnode_function_xid[nid]["id"]+" } .\n"

        elif nid_nodetype_id_questionnode_function_xid[nid]["node_type"] == "literal":
            if nid_nodetype_id_questionnode_function_xid[nid]["type_class"]=="type.datetime":
                id = "'"+nid_nodetype_id_questionnode_function_xid[nid]["id"].split("^^")[0]+"'"+"^^<http://www.w3.org/2001/XMLSchema#datetime>"
            else:
                id=nid_nodetype_id_questionnode_function_xid[nid]["id"].split("^^")[0]
            if nid_nodetype_id_questionnode_function_xid[nid]["function"]=="none":
                sparql += "VALUES " + nid_nodetype_id_questionnode_function_xid[nid]["xid"] + " { " + id + " } .\n"
                node_sentence+="VALUES " + nid_nodetype_id_questionnode_function_xid[nid]["xid"] + " { " + id + " } .\n"

            elif nid_nodetype_id_questionnode_function_xid[nid]["function"]==">=" or \
             nid_nodetype_id_questionnode_function_xid[nid]["function"]=="<=" or \
             nid_nodetype_id_questionnode_function_xid[nid]["function"]=="<" or \
                nid_nodetype_id_questionnode_function_xid[nid]["function"] == ">":
                sparql+="FILTER ( "+nid_nodetype_id_questionnode_function_xid[nid]["xid"]+" "+nid_nodetype_id_questionnode_function_xid[nid]["function"]+" "+id+" ) .\n"
                node_sentence+="FILTER ( "+nid_nodetype_id_questionnode_function_xid[nid]["xid"]+" "+nid_nodetype_id_questionnode_function_xid[nid]["function"]+" "+id+" ) .\n"

            elif nid_nodetype_id_questionnode_function_xid[nid]["function"]=="argmax":
                argmax_nid = nid
            elif nid_nodetype_id_questionnode_function_xid[nid]["function"]=="argmin":
                argmin_nid = nid

    filter_sentences=[]
    nids=list(nid_nodetype_id_questionnode_function_xid.keys())
    for i in range(len(nid_nodetype_id_questionnode_function_xid)):
        for j in range(i+1,len(nid_nodetype_id_questionnode_function_xid)):
            filter_sentences.append(nid_nodetype_id_questionnode_function_xid[nids[i]]["xid"]+" != "+nid_nodetype_id_questionnode_function_xid[nids[j]]["xid"])
    filter_sentence="FILTER ( "+" && ".join(filter_sentences)+ " ) .\n"
    edge_sentence=""
    for edge in grounded_graph.edges:
        edge_sentence += (nid_nodetype_id_questionnode_function_xid[edge.start]["xid"] + " :" + edge.relation + " " +
                          nid_nodetype_id_questionnode_function_xid[edge.end]["xid"] + " .\n")

    if argmax_nid=="" and (argmin_nid)=="":
        sparql+=edge_sentence
    elif (argmax_nid)!="":
        sparql+=("{\n")
        sparql+=("SELECT (MAX("+nid_nodetype_id_questionnode_function_xid[argmax_nid]["xid"].replace("?x","?y")+") AS "+nid_nodetype_id_questionnode_function_xid[argmax_nid]["xid"]+" ) WHERE{\n")
        sparql+=(node_sentence.replace("?x","?y")+edge_sentence.replace("?x","?y"))
        sparql+=(filter_sentence.replace("?x","?y"))
        sparql += ("}\n}\n")
        sparql+=edge_sentence

    elif (argmin_nid)!="":
        sparql += ("{\n")
        sparql += (
        "SELECT (MIN(" + nid_nodetype_id_questionnode_function_xid[argmin_nid]["xid"].replace("?x", "?y") + ") AS " +
        nid_nodetype_id_questionnode_function_xid[argmin_nid]["xid"] + " ) WHERE{\n")
        sparql += (node_sentence.replace("?x", "?y") + edge_sentence.replace("?x", "?y"))
        sparql += (filter_sentence.replace("?x", "?y"))
        sparql += ("}\n}\n")
        sparql += edge_sentence
    sparql+=filter_sentence
    sparql+=("}\n}\n")
    return sparql

