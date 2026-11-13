import os
from ai.hybrid_predict import hybrid_predict

test_image = "media/IMG_1.jpg" if os.path.exists("media/IMG_1.jpg") else "media/test.jpg"

res = hybrid_predict(test_image)
count, visual, model = res[0], res[1], res[2]

print("People :", count)
print("Output :", visual)
print("Model  :", model)