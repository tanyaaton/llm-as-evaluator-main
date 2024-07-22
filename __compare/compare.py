import pandas as pd





# file name
file_name_8b_D8b = [
    'test_D8b_E8bEN_27-06-2024_1405.csv',
    'test_D8b_E8b_25-06-2024_2043.csv',
    'test_D8b_E70bEN_27-06-2024_1413.csv',
    'test_D8b_E70b_25-06-2024_2239.csv',
    'test_D8b_EgptEN_27-06-2024_1428.csv',
    'test_D8b_Egpt_26-06-2024_1044.csv'
]

file_name_70b_D8bTH = [
    'test_D8bTH_E8bEN_27-06-2024_1448.csv',
    'test_D8bTH_E8bTH_26-06-2024_1703.csv',
    'test_D8bTH_E70bEN_27-06-2024_1455.csv',
    'test_D8bTH_E70bTH_27-06-2024_1017.csv',
    'test_D8bTH_EgptEN_27-06-2024_1445.csv',
    'test_D8bTH_EgptTH_27-06-2024_1029.csv'
]

# ans 8b
foldername = 'csv_faithfulness_TH_8b'
filename_human_label = 'lm3_8b_human_label_2_miew.csv'
file_name_list = file_name_8b_D8b

# # ans 70b
# foldername = 'csv_faithfulness_TH_70b'
# filename_human_label = 'lm3_70b_human_label_miew.csv'
# file_name_list = file_name_70b_D8bTH

humanlabel_df = pd.read_csv(f'{foldername}/{filename_human_label}')
humanlabel_mean = round(float(humanlabel_df.loc[:,["faithfulness"]].mean()),3)

result_df = pd.DataFrame(index=['humanlabel_mean','faithfulness_mean','differnce_mean',
                                  'TP', 'TN', 'FP', 'FN','total_question_no',
                                  'TP_percent','TN_percent','FP_percent','FN_percent',])

def tp_tn_fp_fn(compare_df, col_name):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # Iterate over rows
    for index, row in compare_df.iterrows():
        if row['human_label'] == 1 and row[f'{col_name}'] == 1:
            TP += 1
        elif row['human_label'] == 0 and row[f'{col_name}'] == 0:
            TN += 1
        elif row['human_label'] == 0 and row[f'{col_name}'] == 1:
            FP += 1
        elif row['human_label'] == 1 and row[f'{col_name}'] == 0:
            FN += 1
    return TP, TN, FP, FN

with open(f'{foldername}/result_8b.txt', 'w') as f:
    for i in file_name_list:
        data_df = pd.read_csv(f'{foldername}/{i}')
        mean = round(float(data_df.loc[:,["faithfuln"]].mean()),3)
        f.write(i[:-20]+'\n')
        f.write(f'faithfulness mean = {mean}\n')
        difference_mean = round(mean- humanlabel_mean,3)
        f.write(f'difference = {difference_mean}\n')

        # data_df['llama3_ff'] = data_df['llama3_label'].apply(lambda x: x[0]).astype(int)
        data_df['llama3_ff'] = pd.to_numeric(data_df['llama3_label'].apply(lambda x: x[0]), errors='coerce')
        data_df['llama3_ff'].fillna(0, inplace=True)
        data_df['llama3_ff'] = data_df['llama3_ff'].astype(int)

        compare_df = pd.DataFrame()
        compare_df['human_label'] = humanlabel_df['human_label'].astype(int)
        compare_df[f'{i}'] = data_df['llama3_ff']

        TP, TN, FP, FN = tp_tn_fp_fn(compare_df, i)
        f.write(f"True Positives (TP): {TP}\n")
        f.write(f"True Negatives (TN): {TN}\n")
        f.write(f"False Positives (FP): {FP}\n")
        f.write(f"False Negatives (FN): {FN}\n")

        total = TP + TN + FP + FN
        TP_percent = round((TP / total) * 100, 2)
        TN_percent = round((TN / total) * 100, 2)
        FP_percent = round((FP / total) * 100, 2)
        FN_percent = round((FN / total) * 100, 2)

        # f.write results
        total = TP + TN + FP + FN
        f.write(f'out of {total} sub-question\n')
        f.write(f"True Positives (TP) percentage: {TP_percent}%\n")
        f.write(f"True Negatives (TN) percentage: {TN_percent}%\n")
        f.write(f"False Positives (FP) percentage: {FP_percent}%\n")
        f.write(f"False Negatives (FN) percentage: {FN_percent}%\n")
        f.write('\n')

        result_df[i] = {'humanlabel_mean': humanlabel_mean, 'faithfulness_mean': mean, 'differnce_mean': difference_mean,
                                                'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,'total_question_no': total,
                                                'TP_percent': TP_percent,'TN_percent': TN_percent,'FP_percent': FP_percent,'FN_percent': FN_percent}


result_df.to_csv(f'{foldername}/result_8b.csv')
result_df.to_excel(f'{foldername}/excel/result_8b.xlsx')
