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
        <td>semantic matching</td><td>Slot Matching (BERT) Scoring and Path Matching (BERT) Scoring</td>
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
* Root of dataset: default D:/dataset. Note that you can edit it in common/globals_args.py. 


Note that the following files are in baidu wangpan. The extraction code of all files is **kbqa**.


## Common Resources
* [Eight Resources](https://pan.baidu.com/s/1__BBXhEvUuRfqdurofHooQ): GloVe (glove.6B.300d), Stanford CoreNLP server, SUTime Java library, BERT pre-trained Models, and four preprocessing files(stopwords.txt, ordinal_fengli.tsv, unimportantphrase, and unimportantwords). unzip and save in the root.


## Knowledge Bases (KBs)
* [DBpedia (201604 version)](https://pan.baidu.com/s/1byImrmRmOJC-EfYGwvcmOw), for LC-QuAD 1.0
* [Freebase (2013 version)](https://pan.baidu.com/s/1FWwv1R_7JtO_mpk_6pL_TQ), for GraphQuestions
* [Freebase (latest version)](https://pan.baidu.com/s/1CCxljj_yH9S3Y4Zeh6epmw), for ComplexWebQuestions


Note that download a [virtuoso server](http://vos.openlinksw.com/owiki/wiki/VOS) and load the above KBs. The [file](http://ws.nju.edu.cn/blog/2017/03/virtuoso%E5%AE%89%E8%A3%85%E5%92%8C%E5%AF%BC%E5%85%A5%E6%95%B0%E6%8D%AE/) is helpful, if you meet questions.


## LC-QuAD 1.0 Resources
* [LC-QuAD 1.0 dataset](https://pan.baidu.com/s/106vC73W9WKXyuuFcaoPIuQ): Skeleton Parsing models, Word-level scorer model. unzip and save in the root.
* [Lexicons of the corresponding KB (DBpedia 201604)](https://pan.baidu.com/s/1VfF7O0TDRCKiZxqxRpQ8fQ): Entity-related Lexicons and KB schema-related lexicons. unzip and save in the root.

## GraphQuestions Resources
* [GraphQuestions dataset](https://pan.baidu.com/s/106vC73W9WKXyuuFcaoPIuQ): Skeleton Parsing models, Word-level scorer model. unzip and save in the root.
* [Lexicons of the corresponding KB (Freebase 2013)](https://pan.baidu.com/s/1VfF7O0TDRCKiZxqxRpQ8fQ): Entity-related Lexicons and KB schema-related lexicons. unzip and save in the root.

## CWQ 1.1 Resources
* [CWQ 1.1 dataset](https://pan.baidu.com/s/1N_WBCmoQIvNCk_W4oFHeKA): skeleton parsing models, word-level scorer model, sentence-level scorer model. unzip and save in the root.
* [Lexicons of the corresponding KB (Freebase latest)](https://pan.baidu.com/s/146e7C4LCrNiQJp6urZU_ZQ): entity-related lexicons and KB schema-related lexicons. unzip and save in the root.

## Run SkeletonKBQA Pipeline
The pipeline has two KBQA approaches: Skeleton-based Semantic Parsing approach (**SSP**) and Skeleton-based Information Retrieval approach (**SIR**).


## Contacts
If you have any difficulty or questions in running codes, reproducing experimental results, and skeleton parsing, please email to him (ywsun at smail.nju.edu.cn). 



