# %%
import matplotlib.pyplot as plt
import sys
import numpy as np
import pandas as pd
import random
sys.path.append("..")
from src.data import loadData

# %%
classNames = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
trainData, testData = loadData()
print("Training samples:", len(trainData))
print("Test samples:", len(testData))
print("Classes:", classNames)

# %%
plt.figure(figsize=(8, 8))
for i in range(9):
    img, label = trainData[random.randint(0, len(trainData)-1)]
    plt.subplot(3, 3, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(classNames[label])
    plt.axis('off')
plt.show()

# %%
labels = np.array(trainData.targets)
plt.figure(figsize=(8, 4))
plt.bar(range(10), np.bincount(labels))
plt.xticks(range(10), classNames, rotation=45)
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Plot 1 – Target Distribution")
plt.show()

# %%
means = []
labels = []
for image, label in trainData:
    arr = np.array(image)
    means.append(arr.mean())
    labels.append(label)

df = pd.DataFrame({"Label": labels, "Mean Intensity": means})
plt.figure(figsize=(8,4))
df.boxplot(column="Mean Intensity", by="Label", grid=False)
plt.xticks(range(1, 11), classNames, rotation=45)
plt.xlabel("Class")
plt.ylabel("Mean Intensity")
plt.title("Plot 2 – Mean Pixel Intensity per Class")
plt.show()
print("""
EDA Observations:
- Dataset contains 60,000 training and 10,000 test samples.
- All classes are balanced (~6,000 each).
- Pixel values range 0–255(although pixels rarely reach 255, mostly stays below 200); \n mean intensity varies by class.
""")

# %%
