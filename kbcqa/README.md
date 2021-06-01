# SkeletonKBQA: Skeleton Parsing for Complex Question Answering over Knowledge Bases

Codes for a journal paper: "Skeleton Parsing for Complex Question Answering over Knowledge Bases" . 

## Project Structure:

<table>
    <tr>
        <th>File</th><th>Description</th>
    </tr>
    <tr>
        <td>kbcqa</td><td>Codes of skeleton-based SP and IR approaches</td>
    </tr>
    <tr>
        <td>skeletons</td><td>Skeleton Bank</td>
    </tr>
	<tr>
        <td>case study</td><td>Two examples of skeleton-based SP and IR approaches</td>
    </tr>
</table>


## Requirements
* [requirements.txt](https://github.com/nju-websoft/SkeletonKBQA/tree/main/kbcqa/requirements.txt)


## Configuration
The cofiguration of SkeletonKBQA is in common/globals_args.py.
* root: root of all data, default E:/dataset.  
* q_mode: a specific KBQA dataset: lcquad, graphq, and cwq.
* parser_mode: skeleton or dependency.
* sutime: the jar files path of SUTime Java library tool.
* corenlp_ip_port: the ip port of Stanford CoreNLP server.
* dbpedia_pyodbc: the odbc of DBpedia virtuoso server.
* dbpedia_sparql_html: the web site of DBpedia virtuoso server.
* freebase_pyodbc: the odbc of Freebase virtuoso server.
* freebase_sparql_html: the web site of Freebase virtuoso server.


## Common Resources
* Stanford CoreNLP server, SUTime Java library, and BERT pre-trained Models: [Google Drive](https://drive.google.com/file/d/1LZmyVCuW0NPCEapm3l_ztBUK_bWdGEs1/view?usp=sharing). Download, unzip the zip file, and then copy it to the root folder.


## Knowledge Bases
* DBpedia ([201604 version](http://downloads.dbpedia.org/wiki-archive/dbpedia-version-2016-04.html)) for LC-QuAD 1.0 
* Freebase ([2013 version](http://commondatastorage.googleapis.com/freebase-public)) for GraphQuestions
* Freebase ([latest version](https://developers.google.com/freebase)) for ComplexWebQuestions 1.1


Note that download a [virtuoso server](http://vos.openlinksw.com/owiki/wiki/VOS) and load the above KBs. In addition, you do not need to download all KBs. You only need to load a specific KB which is correspond to your KBQA dataset.


## LC-QuAD 1.0
* [Skeleton Parsing models](https://pan.baidu.com/s/106vC73W9WKXyuuFcaoPIuQ): unzip and save in the root.
* [The corresponding KB Lexicons](https://pan.baidu.com/s/1stBDoY6Xdz2d6TeBmq_DJA): Entity-related and KB schema-related lexicons. unzip and save in the root.


## GraphQuestions
* [Skeleton Parsing models](https://pan.baidu.com/s/106vC73W9WKXyuuFcaoPIuQ): unzip and save in the root.
* [The corresponding KB Lexicons](https://pan.baidu.com/s/1VfF7O0TDRCKiZxqxRpQ8fQ): Entity-related and KB schema-related lexicons. unzip and save in the root.


## ComplexWebQuestions 1.1
* [Skeleton Parsing models](https://pan.baidu.com/s/1N_WBCmoQIvNCk_W4oFHeKA): unzip and save in the root.
* [The corresponding KB Lexicons](https://pan.baidu.com/s/146e7C4LCrNiQJp6urZU_ZQ): entity-related and KB schema-related lexicons. unzip and save in the root.


## Run SkeletonKBQA
SkeletonKBQA contains two KBQA approaches:

* Skeleton-based Semantic Parsing approach (**SSP**)
* Skeleton-based Information Retrieval approach (**SIR**)


## Results
We provide results for LC-QuAD 1.0, GraphQuestions, and ComplexWebQuestions 1.1. 


## Contacts
If you have any difficulty or questions in running codes, reproducing experimental results, and skeleton parsing, please email to him (ywsun at smail.nju.edu.cn). 


