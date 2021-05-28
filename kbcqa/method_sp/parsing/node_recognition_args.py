
from skeleton_parsing.skeleton_args import nltk_nlp
from sutime import SUTime
from common import globals_args

wh_words_set = {"what", "which", "whom", "who", "when", "where", "why", "how", "how many", "how large", "how big"}
nltk_nlp = nltk_nlp
sutime = SUTime(jars=globals_args.argument_parser.sutime_jar_files, mark_time_ranges=True)


