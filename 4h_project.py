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
columns = df.columns.to_list()

# Generate an image of the data
# imgdata = img_df.iloc[0, :].to_numpy().reshape((16, 16))
# imgplot = plt.imshow(np.logical_not(imgdata), cmap='gray')
# plt.axis('off')
# plt.show()

def gen_dataset(dataframe):
    copy = dataframe.copy()
    img_df = copy.iloc[:, :256]
    num_df = copy.iloc[:, 256:]
    dataset = tf.data.Dataset.from_tensor_slices((dict(img_df), dict(num_df)))
    return dataset

# Data preprocessing
train_ds = gen_dataset(train)
valid_ds = gen_dataset(valid)
test_ds = gen_dataset(test)

feature_cols = [tf.feature_column.indicator_column(header) for header in columns[:256]]
# feature_layer = tf.keras.layers.DenseFeatures(feature_cols)

# model = tf.keras.Sequential([
#     feature_layer,
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(1)
#     ])

# model.compile()
# print(datasets)
