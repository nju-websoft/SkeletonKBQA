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
        <td>skeletons</td><td>Skeleton Bank from three complex KBQA datasets</td>
    </tr>
	<tr>
        <td>case study</td><td>Two examples of skeleton-based SP and IR approaches</td>
    </tr>
</table>


## Skeleton Bank

We annotate and publish a skeleton bank of 15,166 questions from three KBQA datasets. The skeleton bank is json format, like this:

```
{
"question": "People from the country with the capital Brussels speak what languages ?",
"skeleton": [
	{
		"question": "People from the country with the capital Brussels speak what languages ?",
		"text_span": "with the capital Brussels",
		"headword_index": 3,
		"attachment_relation": "nmod"
	},
	{
		"question": "People from the country speak what languages ?",
		"text_span": "from the country",
		"headword_index": 0,
		"attachment_relation": "nmod"
	}
]
}
```


**Note that we will explain how to run the codes of kbcqa file below.**


## Requirements
* [requirements.txt](https://github.com/nju-websoft/SkeletonKBQA/blob/main/kbcqa/requirements.txt)


## Configuration
The cofiguration of SkeletonKBQA is in kbcqa/common/globals_args.py.
* root: root of all resources and datasets, default ../dataset.  
* q_mode: a specific KBQA dataset: lcquad, graphq, and cwq.
* sutime: jar files path of SUTime Java library tool.
* corenlp_ip_port: ip port of Stanford CoreNLP server.
* dbpedia_pyodbc: odbc of DBpedia virtuoso server.
* dbpedia_sparql_html: web site of DBpedia virtuoso server.
* freebase_pyodbc: odbc of Freebase virtuoso server.
* freebase_sparql_html: web site of Freebase virtuoso server.


## Common Resources
The zip file from [google drive](https://drive.google.com/file/d/1LZmyVCuW0NPCEapm3l_ztBUK_bWdGEs1/view?usp=sharing) contains three parts:
* Stanford CoreNLP server
* SUTime Java library
* BERT pre-trained Models

Note that download, unzip the zip file, and then copy it to the root folder.


## Knowledge Bases
* DBpedia [201604 version](http://downloads.dbpedia.org/wiki-archive/dbpedia-version-2016-04.html) for LC-QuAD 1.0 
* Freebase [2013 version](http://commondatastorage.googleapis.com/freebase-public) for GraphQuestions
* Freebase [latest version](https://developers.google.com/freebase) for ComplexWebQuestions 1.1


Note that download a [virtuoso server](http://vos.openlinksw.com/owiki/wiki/VOS) and load the above KBs. You only need to load a specific KB which is correspond to your KBQA dataset.


## LC-QuAD 1.0 Resources
The zip file from [google drive](https://drive.google.com/file/d/1lpdtISia5HYlVigZ_C5HWPFDhNCerYf_/view?usp=sharing) contains three parts: 
* LC-QuAD 1.0 datasets
* Its skeleton parsing models
* Its corresponding KB entity-related Lexicons

Note that download, unzip the zip file, and then copy it to the root.

## GraphQuestions Resources 
The zip file from [google drive](https://drive.google.com/file/d/1jMf-GyZNEN3Pb1bP2PhoobnSZLFRXSTA/view?usp=sharing) contains three parts: 
* GraphQuestions datasets
* Its skeleton parsing models
* Its corresponding KB entity-related Lexicons

Note that download, unzip the zip file, and then copy it to the root.

## ComplexWebQuestions 1.1 
The zip file from [google drive](https://drive.google.com/file/d/1nzSVhHgozhPO7teY078jtKH42T-fXoUO/view?usp=sharing) contains three parts: 
* ComplexWebQuestions 1.1 datasets
* Its skeleton parsing models
* Its corresponding KB entity-related Lexicons

Note that download, unzip the zip file, and then copy it to the root.

## Run SkeletonKBQA
SkeletonKBQA contains two KBQA approaches:

* Skeleton-based semantic parsing approach (**SSP**) which has four modules:
  - Ungrounded query generation
  - Entity linking
  - Candidate grounded query generation
  - Semantic matching

run the provided SSP script as: 
```
bash run_ssp_LCQ.sh
bash run_ssp_GraphQ.sh
bash run_ssp_CWQ.sh
```

* Skeleton-based Information Retrieval approach (**SIR**) which has three modules:
  - Node recogniztion and linking
  - Candidate grounded path generation
  - Semantic matching


run the provided SIR script as: 
```
bash run_sir_LCQ.sh
bash run_sir_GraphQ.sh
bash run_sir_CWQ.sh
```


## Contacts
If you have any difficulty or questions in running codes, reproducing experimental results, and skeleton parsing, please email to him (ywsun at smail.nju.edu.cn). 


