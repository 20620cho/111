# STEP 1.1: 라이브러리 불러오기 (joblib 추가)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib # 모델 저장을 위해 joblib 사용

# STEP 1.2: 데이터 불러오기 및 모델 학습 (기존 코드 재사용)
df = pd.read_csv("earthquake_data_tsunami.csv")
X = df[["magnitude", "depth", "latitude", "longitude"]]
y = df["tsunami"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# STEP 1.3: 학습된 모델 저장
# 파일명: tsunami_model.joblib
joblib.dump(model, 'tsunami_model.joblib')

print("모델이 'tsunami_model.joblib' 파일로 성공적으로 저장되었습니다.")
