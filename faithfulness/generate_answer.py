import os
from function import ( connect_watsonx_embedding, connect_sentencetransformer_embedding,connect_watsonx_llm, connect_to_milvus, 
                      embedding_data ,find_answer_doc_from_q_df, generate_doc, find_response, drop_milvus_collection)
import pandas as pd


# IBM embedding (en)
# model_id_embedding='ibm/slate-125m-english-rtrvr'
# model_embedding = connect_watsonx_embedding(model_id_embedding)

# SentenceTransformer embedding (th)
model_id_embedding='kornwtp/simcse-model-phayathaibert'
model_embedding = connect_sentencetransformer_embedding(model_id_embedding)

# choose llm IBM model
model_id_llm_llama3='meta-llama/llama-3-70b-instruct'
model_llm_llama3  = connect_watsonx_llm(model_id_llm_llama3)
# milvus-server --proxy-port 19530
connect_to_milvus()

# create question dataframe 
# question_miew_df = pandas.read_csv('csv_files/question.csv')
TH_question_list = [ 'มีลาประเภทใดบ้าง',
'ฉันควรทำอย่างไรถ้าฉันต้องลาเนื่องจากเจ็บป่วย',
'ฉันสามารถลาพักผ่อนได้กี่วัน',
'ฉันสามารถลาไปหนึ่งสัปดาห์แล้วยังได้รับเงินเดือนได้ไหม',
'ฉันสามารถโอนวันลาพักผ่อนที่เหลือจากปีนี้ออกไปไปปีหน้าได้ไหม',
'เงื่อนไขสำหรับการลาคลอดคืออะไร',
'ฉันจะได้รับค่าจ้างในช่วงลาคลอดไหม',
'ฉันสามารถลาคลอดได้เร็วที่สุดเมื่อไร',
'ถ้าฉันทำงานในวันหยุด 2 วันเมื่อเดือนที่แล้ว ฉันสามารถใช้วันหยุดเหล่านั้นในภายหลังได้ไหม',
'ฉันจะขยายระยะเวลาการลาได้อย่างไร',
'ทำไมท้องฟ้าถึงเป็นสีฟ้า']

question_TH_df = pd.DataFrame()
question_TH_df['question'] =  TH_question_list

# embed data
thai_text = open("text/leave_policy_TH.txt", encoding="utf8").read()
collection = embedding_data(thai_text, model_embedding)

# get hts and save doc file 
hits = find_answer_doc_from_q_df(question_TH_df, collection, model_embedding)
generate_doc(question_TH_df, hits)

content_TH_df = question_TH_df.loc[:,['question','contexts']]
find_response(model_llm_llama3, content_TH_df)

print('export csv file')
question_TH_df.to_csv('csv_faithfulness_TH_STTembed_70b/testTH_doc_stt_70b_2.csv')
content_TH_df.to_csv('csv_faithfulness_TH_STTembed_70b/testTH_answer_stt_70b_2.csv')

drop_milvus_collection()