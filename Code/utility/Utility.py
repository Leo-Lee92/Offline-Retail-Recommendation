import csv

class Error(Exception):
    def __str__(self):
        return "허용되지 않는 행동입니다."

def write_csv_list(dir, filename, item_list):
    with open(dir + filename, 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerows(item_list)