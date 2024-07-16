import csv
# 텍스트 파일에서 데이터를 읽습니다.
with open('data_test.txt', 'r') as txt_file:
    lines = txt_file.readlines()

# CSV 파일로 데이터를 씁니다.
with open('test.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    for line in lines:
        # 각 줄을 공백으로 분리하여 리스트로 만듭니다.
        row = line.strip().split()
        # 리스트를 CSV 파일에 씁니다.
        writer.writerow(row)

print("TXT 파일이 CSV 파일로 변환되었습니다.")
# 텍스트 파일에서 데이터를 읽습니다.
with open('data_train.txt', 'r') as txt_file:
    lines = txt_file.readlines()

# CSV 파일로 데이터를 씁니다.
with open('train.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    
    for line in lines:
        # 각 줄을 공백으로 분리하여 리스트로 만듭니다.
        row = line.strip().split()
        # 리스트를 CSV 파일에 씁니다.
        writer.writerow(row)

print("TXT 파일이 CSV 파일로 변환되었습니다.")
