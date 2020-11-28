
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from tqdm import tqdm
import pandas as pd
import csv

def Assciation_rule_mining(df, min_sup, min_num_items, num_collect):
    pos_detail = df["POS_DETAIL"]
    pog_cate = df["POG-카테고리"]
    pog = df["POG"]
    info_item = df["상품정보"]
    info_cate = df["상품-카테고리"]
    pog_df = pd.merge(pog_cate, pog, on = "POG 그룹 ID", how = "left")
    pog_df.rename(columns = {"DIVISION" : "DIV_CD", "DEPT" : "DEPT_CD", "POG 이름_x" : "POG 이름", "POG 이름_y":"POG 그룹 이름"}, inplace = True)
    pog_df.drop(index = 0, inplace = True)
    pog_df.drop(columns = ["사용 유무", "POG 그룹 ID", "POG ID","POG 이름"], inplace = True)
    pog_df.drop_duplicates(inplace = True)
    #info_item.drop(index = 0, inplace = True)
    info_item.rename(columns = {"DIVISION CODE":"DIV_CD", "DEPT CODE" : "DEPT_CD", "SECTION CODE" : "SECTION_CD", "CLASS CODE" : "CLASS_CD"}, inplace= True)
    info_df = pd.merge(info_item, info_cate, on = ["DIV_CD", "DEPT_CD", "SECTION_CD", "CLASS_CD"], how = "left")
    info_df.drop(columns= ["바코드번호", "판매 가격"], inplace= True)
    pog_in_df = pd.merge(info_df, pog_df, on = ["DIV_CD", "DEPT_CD"], how = "left")
    item_df = pd.merge(pos_detail.loc[:,["쇼핑 ID", "자체상품코드"]], pog_in_df, on = "자체상품코드", how = "left")
    item_df.rename(columns= {"DIVISION CODE" : "DIVISION", "DEPT CODE" : "DEPT"}, inplace = True)
    item_df.drop(index= 0, inplace = True)
    item_df.drop_duplicates(inplace = True)
    item_df.reset_index(drop= True, inplace = True)

    #DIV_NAME들을 구분하여 GW카테고리에 맞춰 새로운 칼럼생성
    gw_list = []
    for i in range(len(item_df)):
        name = item_df.loc[i, "DIV_NAME"]
        if name == "과일":
            gw_list.append("청과")
        elif name in ["생선","건해산물"]:
            gw_list.append("수산")
        elif name in ["돼지고기" ,"소고기"]:
            gw_list.append("축산")
        elif name == "조리식품":
            gw_list.append("델리카")
        elif name in ["H&B", "섬유잡화", "장신잡화", "남성의류편집", "피혁잡화", "신발", "아동유아편집"]:
            gw_list.append("H&B")
        elif name in ["음료", "주류"]:
            gw_list.append("음료,주류")
        elif name in ["일반스포츠", "레져스포츠", "스포츠NB"]:
            gw_list.append("스포츠")
        else:
            gw_list.append(name)
    item_df["GW_Cate"] = gw_list
    item_df.drop(columns = ["POG 그룹 이름", "DEPT_NAME","SECTION_NAME", "CLASS_NAME"], inplace= True)
    item_df.drop_duplicates(inplace = True)
    item_df.dropna(inplace = True)
    item_df.reset_index(drop = True, inplace= True)

    shid_list = item_df["쇼핑 ID"].unique()
    transaction_list = []
    for s_id in tqdm(shid_list,  desc = "Create Transaction_Data"):
        transaction_list.append(list(item_df[item_df["쇼핑 ID"] == s_id].GW_Cate.unique()))

    t_encoder = TransactionEncoder()
    t_array = t_encoder.fit(transaction_list).transform(transaction_list)
    t_df = pd.DataFrame(t_array, columns= t_encoder.columns_)

    fre_itemset = apriori(t_df, min_support= min_sup, use_colnames= True)
    #rules = association_rules(fre_itemset, metric="confidence", min_threshold=0.3)
    fre_itemset = apriori(t_df, min_support= min_sup, use_colnames= True)
    fre_itemset.sort_values(by = "support", ascending = False, inplace = True)
    fre_itemset.reset_index(drop = True, inplace= True)

    i = 0
    count = 0
    item_list=[]
    while(True):
        if count == num_collect:
            break
        if len(fre_itemset.itemsets[i]) >= min_num_items:
            count += 1
            item_list.append(list(fre_itemset.itemsets[i]))
        i+=1

    return item_list

