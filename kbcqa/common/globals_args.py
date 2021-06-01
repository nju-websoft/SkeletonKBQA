from common.dataset_name import CWQFileName,GraphqFileName,LCQuADFileName
from common.kb_name import KB_DBpedia_201604, KB_Freebase_Latest, KB_Freebase_en_2013
from argparse import ArgumentParser
from skeleton_parsing.nltk_tools import NLTK_NLP
from sutime import SUTime


def get_args():
    parser = ArgumentParser(description="arguments")
    parser.add_argument('--root', type=str, help='root', default='../dataset')
    parser.add_argument('--q_mode', type=str, help='lcquad, cwq, graphq', default='lcquad')
    parser.add_argument('--sutime', type=str, help='SUTime tool', default='../dataset/resources_sutime/python-sutime-master/jars')
    parser.add_argument('--corenlp_ip_port', type=str, help='CoreNlP', default='http://114.212.190.19:9004/')
    parser.add_argument('--dbpedia_pyodbc', type=str, help='LC-QuAD', default='DSN=knowledgebase;UID=dba;PWD=dba')
    parser.add_argument('--dbpedia_sparql_html', type=str, help='LC-QuAD', default="http://114.212.84.164:8890/sparql")
    parser.add_argument('--freebase_pyodbc', type=str, help='CWQ, GraphQ', default='DSN=freebaselatest;UID=dba;PWD=dba')
    parser.add_argument('--freebase_sparql_html', type=str, help='CWQ, GraphQ', default="http://114.212.81.7:8894/sparql")
    return parser.parse_args()


argument_parser = get_args()

fn_lcquad_file = LCQuADFileName(root=argument_parser.root)
fn_graph_file = GraphqFileName(root=argument_parser.root)
fn_cwq_file = CWQFileName(root=argument_parser.root)

kb_dbpedia_201604_file = KB_DBpedia_201604(root=argument_parser.root)
kb_freebase_en_2013 = KB_Freebase_en_2013(root=argument_parser.root)
kb_freebase_latest_file = KB_Freebase_Latest(root=argument_parser.root)

nltk_nlp = NLTK_NLP(argument_parser.corenlp_ip_port)
sutime = SUTime(jars=argument_parser.sutime, mark_time_ranges=True)

