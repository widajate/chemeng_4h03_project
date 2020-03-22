import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Import data
df = pd.read_excel("semeion_data.xlsx")
train, test = train_test_split(df, test_size=0.2)
train, valid = train_test_split(train, test_size=0.2)
columns = df.columns

# Generate an image of the data
# imgdata = img_df.iloc[0, :].to_numpy().reshape((16, 16))
# imgplot = plt.imshow(np.logical_not(imgdata), cmap='gray')
# plt.axis('off')
# plt.show()

def gen_dataset(dataframe):
    copy = dataframe.copy()
    img_df = copy.iloc[:, :256]
    num_df = copy.iloc[:, 256:]
    img_ds = img_df.values
    num_ds = num_df.values
    return img_ds, num_ds

# Data preprocessing
train_x, train_y = gen_dataset(train)
valid_x, valid_y = gen_dataset(valid)
test_x, test_y = gen_dataset(test)

opt = tf.keras.optimizers.SGD(learning_rate=0.05)
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_dim=256),
    tf.keras.layers.Dense(10, activation='softmax')
    ])

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
model.fit(x=train_x, y=train_y, epochs=50, validation_data=(valid_x, valid_y))
score = model.evaluate(test_x, test_y)
print(score)
