
from Association_rule_mining import Assciation_rule_mining
import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utility.Utility import write_csv_list

if __name__ == "__main__":
    #It requires data set
    filename = 'RFID_IPS_데이터추출_20200904.xlsx'
    dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/Data/"
    save_dir =  os.path.dirname(__file__) + "/Data/"
    df = pd.read_excel(dir+filename, sheet_name= ["POS_DETAIL", "POG-카테고리", "POG", "상품정보", "상품-카테고리"])
    rules = Assciation_rule_mining(df, 0.05, 5, 10)
    write_csv_list(save_dir, "rules.csv", rules)
