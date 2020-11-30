#%%
df = pd.read_excel('/home/shinjk1156/Project_code/Retail_Project/Data/RFID_IPS_데이터추출_20200904.xlsx', sheet_name= None)
line = df['라인정보']
repeater = df["리피터 마스터"]
pog = df['POG']
pog_cate = df['POG-카테고리']
shoping = df['가공된 쇼핑 데이터']
#pog.drop(index= 0, inplace = True)
#repeater.drop(index= 0, inplace = True)
pog["POG 그룹 ID"] = pog["POG 그룹 ID"].map(str)
pog_cate["POG 그룹 ID"] = pog_cate["POG 그룹 ID"].map(str)
moving1 = pd.read_csv("/home/shinjk1156/Project_code/Retail_Project/Data/실시간저장데이터_20110208_20110220.txt", sep = "\t", header = None)
moving2 = pd.read_csv("/home/shinjk1156/Project_code/Retail_Project/Data/실시간저장데이터_20110117_20110207.txt", sep = "\t", header = None)
moving = moving1.append(moving2)
moving.columns = ["SINK ID", "태그 ID","TAG 강도", "리피터 ID", "저장시간"]

repeater["리피터 ID"] = repeater["리피터 ID"].astype(str)
moving["리피터 ID"] = moving["리피터 ID"].astype(str)

mer_df = pd.merge(repeater, pog, on = "POG 그룹 ID", how = "left")
to_shop = pd.DataFrame()
# this process take many time,
for i in tqdm(range(1,len(shoping))):
    sample = moving[(moving.저장시간 >= shoping.loc[i,"쇼핑 시작시간"]) & (moving.저장시간 <= shoping.loc[i,"쇼핑 종료 시간"]) & (moving["태그 ID"] ==shoping.loc[i, "태그 ID"])].copy()
    if len(sample) == 0:
        pass
    else:
        sample.sort_values(by = "저장시간", inplace = True)
        sample = pd.merge(sample, repeater, on = "리피터 ID", how = "left")
        sample["쇼핑 ID"] = [shoping.loc[i, "쇼핑 ID"]]*len(sample)
        to_shop = to_shop.append(sample)
#to_shop.to_csv("동선_전처리_0222_0220.csv", index = False)
#%%