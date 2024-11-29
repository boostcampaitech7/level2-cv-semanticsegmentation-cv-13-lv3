import pandas as pd

# Input 파일 경로
palm_file_path = "/data/ephemeral/home/kjh/level2-cv-semanticsegmentation-cv-13-lv3/ensemble/checkpoint/csv/palm/fisrt_palm.csv"
normal_file_path = "/data/ephemeral/home/kjh/level2-cv-semanticsegmentation-cv-13-lv3/ensemble/checkpoint/csv/full/UPerNet_HRNet.csv"

# Output 파일 경로
aligned_csv_path = "./aligned_palm.csv"

# CLASSES와 PALM_CLASSES 정의
CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

PALM_CLASSES = [
    'Hamate', 'Scaphoid', 'Lunate', 'Trapezium', 'Capitate', 
    'Triquetrum', 'Trapezoid', 'Pisiform'
]

# CSV 로드
palm_file_df = pd.read_csv(palm_file_path)
normal_file_df = pd.read_csv(normal_file_path)

# image_name 리스트 가져오기
image_names = normal_file_df["image_name"].unique()

# 빈 데이터프레임 생성
aligned_data = {
    "image_name": [],
    "class": [],
    "rle": []
}

# CLASSES에 따라 정렬된 데이터프레임 생성
for image_name in image_names:
    for cls in CLASSES:
        aligned_data["image_name"].append(image_name)
        aligned_data["class"].append(cls)
        # palm_file_df에서 해당 image_name과 class가 있으면 rle 추가, 없으면 빈칸
        matching_row = palm_file_df[
            (palm_file_df["image_name"] == image_name) & 
            (palm_file_df["class"] == cls)
        ]
        if not matching_row.empty:
            aligned_data["rle"].append(matching_row.iloc[0]["rle"])
        else:
            aligned_data["rle"].append("")  # 누락된 클래스는 빈칸 처리

# DataFrame으로 변환
aligned_df = pd.DataFrame(aligned_data)

# CSV 저장
aligned_df.to_csv(aligned_csv_path, index=False)
print(f"Aligned CSV saved to {aligned_csv_path}")