from common import globals_args
from common import hand_files

parser_mode = globals_args.parser_mode

unimportantwords = hand_files.read_set(globals_args.argument_parser.unimportantwords)
unimportantphrases = hand_files.read_list(globals_args.argument_parser.unimportantphrases)
stopwords_dict = hand_files.read_set(globals_args.argument_parser.stopwords_dir)


