import pickle
from myapp.my_models import EfficientNetEmbeddingModel

# 모델 인스턴스 생성
model = EfficientNetEmbeddingModel()

# 모델을 pkl 파일로 저장
with open('efficientnet_embedding_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully.")
