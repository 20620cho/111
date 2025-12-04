# 파일명: model_train_and_save.py (로컬 환경에서 한 번 실행)

# STEP 1. 라이브러리 불러오기
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib # 모델 저장을 위해 joblib 추가
# !pip install koreanize-matplotlib # 로컬 환경에서 matplotlib 한글 지원 필요 시 주석 해제
# import koreanize_matplotlib

# STEP 2. 데이터 불러오기
try:
    df = pd.read_csv("earthquake_data_tsunami.csv")
except FileNotFoundError:
    print("오류: 'earthquake_data_tsunami.csv' 파일을 찾을 수 없습니다. 파일이 현재 디렉토리에 있는지 확인하세요.")
    exit()

# STEP 3. 필요한 열 선택
X = df[["magnitude", "depth", "latitude", "longitude"]]   # 입력 변수
y = df["tsunami"]

# STEP 4. 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 5. 모델 학습
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 6. 예측 및 평가 (콘솔 출력)
y_pred = model.predict(X_test)
print("--- 모델 평가 결과 ---")
print("정확도:", accuracy_score(y_test, y_pred))
print("\n분류 리포트:\n", classification_report(y_test, y_pred))
print("\n혼동행렬:\n", confusion_matrix(y_test, y_pred))
print("-----------------------")

# STEP 7. 중요 변수 시각화 (matplotlib)
# importances = model.feature_importances_
# plt.figure(figsize=(8, 5))
# plt.bar(X.columns, importances)
# plt.title("Feature Importance (특성이 쓰나미 예측에 미치는 영향)")
# plt.show()

# STEP 8. 학습된 모델 저장
joblib.dump(model, 'tsunami_model.joblib')
print("\n✅ 모델이 'tsunami_model.joblib' 파일로 성공적으로 저장되었습니다. 웹앱 실행 준비 완료.")
