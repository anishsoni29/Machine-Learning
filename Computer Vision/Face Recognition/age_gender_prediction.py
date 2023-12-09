import os
import cv2
import pandas as pd
from deepface import DeepFace

data = {"Name": [], "Age": [], "Gender": [], "Race": []}

for file in os.listdir("images"):
    results = DeepFace.analyze(
        cv2.imread(f"images/{file}"), actions=("age", "gender", "race")
    )
    data["Name"].append(file.split(".")[0])
    data["Age"].append(results["age"])
    data["Gender"].append(results["gender"])
    data["Race"].append(results["race"])

df = pd.DataFrame(data)
print(df)

df.to_csv("people.csv", index=False)
