import pickle
from train import NgramModel

loaded_model = pickle.load(open('model.pkl', 'rb'))
result = loaded_model.generate(20)
print(result)