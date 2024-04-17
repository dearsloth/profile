""" 一、資料前處理 : 將病歷斷句 """

#n2c2-句子
import pandas as pd
import os
import re

def extract_sent(text_path,n2c2_drug_path,save_to):
    drug=pd.read_excel(n2c2_drug_path)
    drug_list=[]
    sent_list=[]
    #把drug檔案存入list
    for i in drug.values:
        detail=[]
        detail.append(i[1])
        detail.append(i[2])
        detail.append(i[3])
        detail.append(i[4])
        drug_list.append(detail)        
    #讀取drug_list、text提取句子
    for filename in sorted(os.listdir(text_path)):
        if filename.split('.')[1]=="txt":
            file=open(text_path+filename,'r').read()
            for drug in drug_list:
                if drug[0]==filename.split('.')[0]:
                    #print(drug)
                    text_drug=file[int(drug[2]):int(drug[3])]
                    detail=[]
                    if '+' in drug[1]: 
                        text_drug=text_drug.replace('+','\+')
                    if '(' in drug[1]:
                        text_drug=text_drug.replace('(','\(')
                    if ')' in drug[1]:
                        text_drug=text_drug.replace(')','\)')
                    if '.' in file[int(drug[3]):] and re.search(r''+text_drug+'.*\.',text_drug)!=None:
                        sent=re.findall(r'[^\.]*'+text_drug+'.*\.',file)
                    elif '.' in file[int(drug[3]):]:
                        sent=re.findall(r'[^\.]*'+text_drug+'[^.]*\.',file)
                    else:
                        sent=re.findall(r'[^\.]*'+text_drug+'.*',file)
                    detail.append(drug[0])
                    detail.append(drug[1])
                    detail.append(sent[0])
                    sent_list.append(detail)
    #for sent in sent_list:
        #print(sent)
    df=pd.DataFrame(sent_list)
    df.columns=['filename','drug','sentence'] 
    df.to_excel(save_to, index=False)
    #print(df)
	
dev_text="D:\\NAS\\han\\n2c2\\data\\dev\\"
#train_text="D:\\NAS\\han\\n2c2\\data\\train\\"
n2c2_drug="D:\\NAS\\han\\n2c2\\preprocess\\drugsname\\n2c2_extract_drug.xlsx"
dev_snet_save="D:\\NAS\\han\\n2c2\\task1\\label\\dev_sent.xlsx"
#train_snet_save="D:\\NAS\\han\\n2c2\\task1\\label\\train_sent.xlsx"

extract_sent(dev_text,n2c2_drug,dev_snet_save)




""" 二、使用spaCy套件訓練模型 """

import spacy
from spacy.tokens import DocBin
import srsly
import typer
import warnings
from pathlib import Path
import os
import pandas as pd
import json

#(1)準備input   
#Data放入模型的格式 = [["文本",{"entities":[開始位置,結束位置,"DrugName"]}]]
"""
#若希望input放入整份文本：
def input_text(data_path,drug_path):
    nlp = spacy.load("en_core_web_sm")
    patterns=[]   # =[{"label": "DrugName", "pattern": "Lipitor"}, ...]
    TRAIN_DATA = []  # =[['句1', {'entities': [80,89,'DrugName']}],['句2', {'entities': [55,62,'DrugName']}]] 
    
    #讀取藥物名稱 >> 存入drug_list >> 存入patterns
    df = pd.read_excel(drug_path, sheet_name="Sheet1",usecols=["drug"])
    drug_list=[]
    for I in df.values:
        for i in I:
            drug_list.append(i)
    for drug in drug_list:
        label={}
        label["label"]="DrugName"
        label["pattern"]=drug
        patterns.append(label)
    
    #把文本存入list
    text_list=[]
    for file in sorted(os.listdir(data_path)):
        if file.split('.')[1]=="txt":
            text=open(data_path+file,'r').read()
            doc = nlp(text)
            text_list.append(doc)
            entities = []
            for ent in doc.ents:
                entities.append([ent.start_char, ent.end_char, ent.label_])
            TRAIN_DATA.append([text, {"entities": entities}])
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    # for x in TRAIN_DATA:
        # print(x)
    # print(len(TRAIN_DATA))
    return TRAIN_DATA
"""

#若希望input放入句子： 
def input_sent(data_path,drug_path,save_label,save_input):
    nlp = spacy.load("en_core_web_sm")
    patterns=[]   # =[{"label": "DrugName", "pattern": "Lipitor"}, ...]
    TRAIN_DATA = []  # =[['句1', {'entities': [80,89,'DrugName']}],['句2', {'entities': [55,62,'DrugName']}]] 
    
    #讀取藥物名稱 >> 存入drug_list >> 存入patterns
    df = pd.read_excel(drug_path, sheet_name="Sheet1",usecols=["drug"])
    drug_list=[]
    for I in df.values:
        for i in I:
            drug_list.append(i)
    for drug in drug_list:
        label={}
        label["label"]="DrugName"
        label["pattern"]=drug
        patterns.append(label)       
    
    #把文本分句存入list
    sent_list=[]
    count_label=0
    for file in sorted(os.listdir(data_path)):
        if file.split('.')[1]=="txt":
            text=open(data_path+file,'r').read()
            doc = nlp(text)
            for sent in doc.sents:
                sent_list.append(sent.text)  
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler")
    ruler.add_patterns(patterns)
    for sentence in sent_list:
        doc = nlp(sentence)
        entities = []
        for ent in doc.ents:
            count_label+=1
            entities.append([ent.start_char, ent.end_char, ent.label_])
        if len(entities)>0:
            TRAIN_DATA.append([sentence, {"entities": entities}])
    #print(count_label)
    #print(len(patterns))
    label = json.dumps(patterns, indent = 4)
    f1 = open(save_label, 'w')
    f1.write(label)
    f1.close()
    input_data = json.dumps(TRAIN_DATA, indent = 4)
    f2 = open(save_input, 'w')
    f2.write(input_data)
    f2.close()
    #return TRAIN_DATA

#==========================================

#(2)輸出成spacy2 (製作train、dev)
#lang => 是空白模型的語言，英語使用“en”
#TRAIN_DATA => 是訓練資料做成列表。
#output_path => spaCy2文件的輸出目錄。
def convert(lang: str, TRAIN_DATA, output_path: Path):
    nlp = spacy.blank(lang)
    db = DocBin()
    for text, annot in TRAIN_DATA:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annot["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                msg = f"Skipping entity [{start}, {end}, {label}] in the following text because the character span '{doc.text[start:end]}' does not align with token boundaries:\n\n{repr(text)}\n"
                warnings.warn(msg)
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(output_path)


#生成input格式
train="D:\\NAS\\han\\n2c2\\data\\train\\"
dev="D:\\NAS\\han\\n2c2\\data\\dev\\"
test="D:\\NAS\\han\\n2c2\\data\\test\\"
#drug="D:\\NAS\\han\\n2c2\\preprocess\\drugsname\\n2c2_extract_drug.xlsx"
drug="D:\\NAS\\han\\n2c2\\preprocess\\drugsname\\n2c2_train_extract_drug.xlsx"
#drug="D:\\NAS\\han\\n2c2\\preprocess\\drugsname\\extradata_simply_extract_drug.xlsx"
#drug="D:\\NAS\\han\\n2c2\\preprocess\\drugsname\\combine_nodev.xlsx"
label_save="D:\\NAS\\han\\n2c2\\task1\\NER_model\\sent_n2c2_NoneDev\\label.json"
input_save="D:\\NAS\\han\\n2c2\\task1\\NER_model\\sent_n2c2_NoneDev\\input.json"
#input_text(train,drug)
#input_text(dev,drug)
#input_sent(train,drug)
#input_sent(dev,drug)
#input_sent(test,drug)
input_sent(train,drug,label_save,input_save)

#生成spacy2模型
#save_train="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\text\\train.spacy"
#save_dev="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\text\\dev.spacy"
#save_train="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\sent\\sent_train.spacy"
#save_dev="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\sent\\sent_dev.spacy"
#save_test="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\sent\\sent_test.spacy"
#save_train="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\sent_LabelNoneDev\\sent_train.spacy"
#save_dev="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\sent_LabelNoneDev\\sent_dev.spacy"
#save_test="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\sent_LabelNoneDev\\sent_test.spacy"
save_train="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\sent_ExtraData\\sent_train.spacy"
save_dev="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\sent_ExtraData\\sent_dev.spacy"
save_test="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\sent_ExtraData\\sent_test.spacy"
#convert("en", input_text(train,drug), save_train)
#convert("en", input_text(dev,drug), save_dev)
#convert("en", input_sent(train,drug), save_train)
#convert("en", input_sent(dev,drug), save_dev)
#convert("en", input_sent(test,drug), save_test)



#創建spaCy config.cfg (設定參數)
#至此網址配置合適語法https://spacy.io/usage/training
#創建好自己的base_config.cfg(存在與上面函式save相同路徑)，於cmd執行下面程式
#python -m spacy init fill-config D:\NAS\han\n2c2\task1\NER_Model\text\base_config.cfg D:\NAS\han\n2c2\task1\NER_Model\text\config.cfg

#出現下面兩句：(執行第二句，要改成自己的路徑)
#You can now add your data and train your pipeline:
#python -m spacy train config.cfg --paths.train ./train.spacy --paths.dev ./dev.spacy

#python -m spacy train D:\NAS\han\n2c2\task1\NER_Model\text\config.cfg --paths.train D:\NAS\han\n2c2\task1\NER_Model\text\train.spacy --paths.dev D:\NAS\han\n2c2\task1\NER_Model\text\dev.spacy --output D:\NAS\han\n2c2\task1\NER_Model\text\output
#python -m spacy train D:\NAS\han\n2c2\task1\NER_Model\sent\config.cfg --paths.train D:\NAS\han\n2c2\task1\NER_Model\sent\sent_train.spacy --paths.dev D:\NAS\han\n2c2\task1\NER_Model\sent\sent_dev.spacy --output D:\NAS\han\n2c2\task1\NER_Model\sent\output
#python -m spacy train D:\NAS\han\n2c2\task1\NER_Model\text\config.cfg --paths.train D:\NAS\han\n2c2\task1\NER_Model\text\train.spacy --paths.dev D:\NAS\han\n2c2\task1\NER_Model\text\dev.spacy


#======================================================
#轉換成spacy3
#執行：
#python -m spacy train data/config.cfg --output ./models/output

#python -m spacy train D:\NAS\han\n2c2\task1\NER_Model\sent\config.cfg --output D:\NAS\han\n2c2\task1\NER_Model\sent\output


#測試模型
def text_model(model_path,test_data_path):
    trained_nlp = spacy.load(model)
    entities_dic={}
    for file in sorted(os.listdir(test_data_path)):
        if file.split('.')[1]=='txt':
            text=open(test_data_path+file,'r').read()
            doc = trained_nlp(text)          
            detail=[]
            for ent in doc.ents:
                detail.append(ent.text)
            entities_dic[file.split('.')[0]]=detail    
                #print(ent.text)
                #print(text.index(ent.text))
            #if len(doc.ents) == 0:
                #print("No entities found.")
    print(entities_dic)            
             

model="D:\\NAS\\han\\n2c2\\task1\\NER_Model\\sent\\output\\model-best"
test="D:\\NAS\\han\\n2c2\\data\\test\\"
text_model(model,test) 




""" 三、繪製曲線圖 """

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 

#曲線圖
def line_chart(excel):
    df = pd.read_excel(excel, sheet_name='spacy_extradata')
    #print(df)
    sns.set() 
    sns.lineplot(x="Epoch",y="range",hue='index',data=df)
    plt.show()

spacy="D:\\NAS\\han\\n2c2\\task1\\score.xlsx"
line_chart(spacy)