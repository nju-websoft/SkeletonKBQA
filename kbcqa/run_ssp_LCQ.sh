python method_sp/sp_pipeline.py  \
        --q_mode  lcquad  \
        --parser_mode  skeleton  \
        --module  1_ungrounded_query_generation  \
        --dataset test \
        --output_folder output_lcquad_e2e  \
        --output_configuration  output_lcquad_sp_skeleton_slot_e2e_sp2  \
        --ungrounded_file  structures_with_1_ungrounded_graphs_test_skeleton.json  \
        --_2_1_grounded_file  structures_with_2_1_grounded_graph_test_skeleton.json  \
        --grounded_folder  2.2_test  \
        --output_result_file  2021.02.24_output_LCQuAD_SP_2_E2E.json

