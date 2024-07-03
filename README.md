# 📊 LLM as evaluator

This lab impliment watsonx llm model to use as evaluator for RAG application

## Faithfulness

### 📌 generate_answer.py
This function generates answer from your input question list, using model of your choice
1. store your text file, which contain input for the RAG application, in folder `text`
2. store your question file that contain list of question related to the RAG document in folder `csv_files`
3. choose the llm and embedding model you want to evaluate in the `utils/config` file, input the detail of the model in `generate answer`
4. run **milvus lite** on your local device by running the following command in terminal
`milvus-server --proxy-port 19530`

After run generate_answer.py, `content.csv` will be generated in folder `csv_files`
The file contains given **question, answer generate from chosen model, and context support the answer**

### 📌 faithfulness.py
This function generates answer from generate_answer.py is stored in the your input question list, using model of your choice
1. make sure the content file from store your content file, which contain column **question, answwr, context** in folder `csv_files`
3. choose the llm you want to use in the evaluation process in the `utils/config` file, input the detail of the model in `faithfulness`

After run faithfulness.py, `eval_faithfulness_xx-xx-xxxx_xxxx.csv` and `eval_faithfulness_detail_xx-xx-xxxx_xxxx.csv` will be generated in folder `csv_files`
providing faithfulness score for the answer generate from chosen model and it's detail repectively
The file contains given **question, answer generate from chosen model, context support the answer, and faithfulness file**
