#%%
import pandas as pd 
from datetime import datetime
from tqdm import tqdm
import numpy as np
#%%
def correction(trans_list):
    count_list = []
    shop_list = []
    i = 0
    j = 1
    count = 1
    while(True):
        if i + j  > len(trans_list)-1:
            count_list.append(count)
            shop_list.append(trans_list[i])
            break
        if trans_list[i] == trans_list[i+j]:
            count += 1
            j +=1
        else:
            count_list.append(count)
            shop_list.append(trans_list[i])
            count = 1
            i = i + j
            j = 1
    return count_list, shop_list
#%%
df = pd.read_excel('/home/shinjk1156/Project_code/Retail_Project/Data/RFID_IPS_데이터추출_20200904.xlsx', sheet_name= None)
fdf = pd.read_csv("/home/shinjk1156/Project_code/Retail_Project/Code/Preprocessing/Data/동선_전처리_0122_0220_final.csv")
#%%
fdf.drop(columns = ["사용 강도", "상품 여부","가중치","리포트에 보여줄지에 대한 여부","스타트 여부","사용여부","POS 여부", "TAG 강도"], inplace = True)
fdf.dropna(inplace = True)
info_item = df["상품정보"]
info_cate = df["상품-카테고리"]
info_item.rename(columns = {"DIVISION CODE":"DIV_CD", "DEPT CODE" : "DEPT_CD", "SECTION CODE" : "SECTION_CD", "CLASS CODE" : "CLASS_CD"}, inplace= True)
info_df = pd.merge(info_item, info_cate, on = ["DIV_CD", "DEPT_CD", "SECTION_CD", "CLASS_CD"], how = "left")
info_df.drop(columns= ["바코드번호", "판매 가격"],index = 0 ,inplace= True)
pog = df['POG']
pog_cate = df['POG-카테고리']
pog["POG 그룹 ID"] = pog["POG 그룹 ID"].map(str)
pog_cate["POG 그룹 ID"] = pog_cate["POG 그룹 ID"].map(str)
pog_df = pd.merge(pog_cate, pog, on="POG 그룹 ID", how = "left")
pog_df.rename(columns = {"DIVISION" : "DIV_CD", "DEPT" : "DEPT_CD", "POG 이름_x" : "POG 이름", "POG 이름_y":"POG 그룹 이름"}, inplace = True)
pog_df.drop(index = 0, inplace = True)
pog_df.drop(columns = ["사용 유무", "POG ID","POG 이름","DEPT_CD"], inplace = True)
pog_df.drop_duplicates(inplace = True)
pog_in_df = pd.merge( pog_df, info_df.loc[:,["DIV_CD", "DIV_NAME"]], on = ["DIV_CD"], how = "left")
pog_in_df.drop(columns = ["POG 그룹 이름"], inplace = True)
pog_in_df.drop_duplicates(inplace = True)
pog_in_df.reset_index(drop = True, inplace = True)

#DIV_NAME들을 구분하여 GW카테고리에 맞춰 새로운 칼럼생성
gw_list = []
for i in range(len(pog_in_df)):
    name = pog_in_df.loc[i, "DIV_NAME"]
    if name == "과일":
        gw_list.append("청과")
    elif name in ["생선","건해산물"]:
        gw_list.append("수산")
    elif name in ["돼지고기" ,"소고기"]:
        gw_list.append("축산")
    elif name == "조리식품":
        gw_list.append("델리카")
    elif name in ["H&B", "섬유잡화", "장신잡화", "남성의류편집", "피혁잡화", "신발", "이지캐주얼" , "아동유아편집","여성의류편집","여성의류NB","남성의류NB","아동유아NB"]:
        gw_list.append("H&B")
    elif name in ["음료", "주류"]:
        gw_list.append("음료,주류")
    elif name in ["일반스포츠", "레져스포츠", "스포츠NB"]:
        gw_list.append("스포츠")
    else:
        gw_list.append(name)

pog_in_df["GW_Cate"] = gw_list
pog_in_df.drop(columns = ["DIV_CD", "DIV_NAME"],inplace = True)
pog_in_df.drop_duplicates(inplace = True)

pog_id = []
gw_ca = []
for i in range(17):
    if i == 0:
        pog_id.append(0)
        gw_ca.append("입구")
    else:
        pog_id.append(i)
        gw_ca.append("계산대")
io_df = pd.DataFrame()
io_df["POG 그룹 ID"] = pog_id
io_df["GW_Cate"] = gw_ca

pog_in_df = pd.concat([pog_in_df, io_df])
pog_in_df["POG 그룹 ID"] = pog_in_df["POG 그룹 ID"].astype(int)
fdf["POG 그룹 ID"] = fdf["POG 그룹 ID"].astype(int)
final = pd.merge(fdf, pog_in_df, on= "POG 그룹 ID", how = "left")
#final.to_csv("동선정보_매핑완료_0122_0220_final.csv", index = False)

#data = pd.read_csv('/home/shinjk1156/Project_code/Retail_Project/Code/Preprocessing/Data/동선정보_매핑완료_0122_0220_final.csv')
data = final.copy()
transactions = []
for id in tqdm(data["쇼핑 ID"].unique()):
    temp = data[data["쇼핑 ID"] == id]
    temp.reset_index(inplace = True, drop = True)
    d_index_list = []
    for i in range(1,len(temp)-1,1):
        if (temp.loc[i,"GW_Cate"] != temp.loc[i+1,"GW_Cate"]) and (temp.loc[i,"GW_Cate"]!= temp.loc[i-1,"GW_Cate"]):
            d_index_list.append(i)
        else:
            pass
    temp.drop(index = d_index_list, inplace = True)
    temp.reset_index(drop = True, inplace  = True)
    cor_list = correction(temp.GW_Cate.to_list()) #횟수, 동선리스트
    i = 0
    f_list = []
    for cor in range(len(cor_list[0])):
        #print(cor_list[1][cor])
        if cor == 0:
            start = datetime.strptime(str(temp.loc[i,"저장시간"]), "%Y-%m-%d %H:%M:%S.%f")
            #print(str(temp.loc[i,"저장시간"]))
            j = cor_list[0][cor]
            i = i + j -1
            end = datetime.strptime(str(temp.loc[i,"저장시간"]), "%Y-%m-%d %H:%M:%S.%f")
            #print(str(temp.loc[i,"저장시간"]))
        else:
            start = datetime.strptime(str(temp.loc[i+1,"저장시간"]), "%Y-%m-%d %H:%M:%S.%f")
            #print(str(temp.loc[i+1,"저장시간"]))
            j = cor_list[0][cor]
            i = i + j 
            end = datetime.strptime(str(temp.loc[i,"저장시간"]), "%Y-%m-%d %H:%M:%S.%f")
            #print(str(temp.loc[i,"저장시간"]))
        f_list = f_list + [cor_list[1][cor]]*(int((end-start).total_seconds()/60) + 1)
    transactions.append(f_list)


trans_dic = {}
count = 0
for trans in transactions:
    for element in trans:
        if element not in trans_dic.keys():
            trans_dic[element] = count
            count += 1
        else:
            pass
#동선정보에 기록이 안된 매대(리피터 오류로 추정됨)
out = ["H&B", "곡물","언더웨어","스포츠","문구"]
for h in out:
    trans_dic[h] = count
    count += 1

n_trans_list = []
for trans in transactions:
    t_list = []
    for element in trans:
        t_list.append(str(trans_dic[element]))
    n_trans_list.append(t_list)

trans_matrix = pd.DataFrame(np.zeros((len(trans_dic.keys()), len(trans_dic.keys()))),index = trans_dic.keys(), columns= trans_dic.keys())
#인접지역 입력 "H&B", "곡물","언더웨어","스포츠","문구"
c_state = {"H&B" : ["곡물", "계산대","스포츠","문구","완구","자동차","언더웨어","청소욕실"],
"곡물" : ["청과","계산대","채소","H&B"],
"언더웨어" : ["H&B","애완원예용품","인테리어","건강,차","청소욕실"],
"스포츠" : ["H&B","문구"], "문구" : ["스포츠", "H&B", "완구"]}

for t_list in transactions:
    for t in range(len(t_list)-1):
        trans_matrix.loc[t_list[t], t_list[t+1]] += 1

for state in c_state.keys():
    temp = c_state[state]
    for s in temp:
        trans_matrix.loc[state,s] = 1/len(temp)

for inx in trans_matrix.index:
    #소수점 3번째 이하 0으로 처리
    trans_matrix.loc[inx, :] = round(trans_matrix.loc[inx, :] / sum(trans_matrix.loc[inx, :]),2)
    
    for jix in trans_matrix.index:
        if trans_matrix.loc[inx, jix] < 0.03: 
            trans_matrix.loc[inx, jix] = 0

for state in c_state.keys():
    temp = c_state[state]
    for s in temp:
        if s in ["문구","완구","자동차","애완원예용품","인테리어"]:
            trans_matrix.loc[s,state] = 1/3
        elif s in ["스포츠","청과","계산대"]:
            trans_matrix.loc[s,state] = 1/2
        elif s in ["곡물","청소욕실","건강,차"]:
            trans_matrix.loc[s,state] = 1/4
        elif s == "언더웨어":
            trans_matrix.loc[s,state] = 1/5
        elif s == "채소":
            trans_matrix.loc[s,state] = 1/6
        else:
            trans_matrix.loc[s,state] = 1/8


for inx in trans_matrix.index:
    #다시 각 row의 합이 1이 되게 설정함
    trans_matrix.loc[inx, :] = trans_matrix.loc[inx, :] / sum(trans_matrix.loc[inx, :])

trans_matrix.to_csv("transition_probability_final.csv")
trans_matrix


# %%
