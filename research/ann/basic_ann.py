import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import keras


class BasicAnn:
    X = 0

    def __init__(self):
        self.dataset = None
        self.x = None
        self.y = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.ann = None
        self.label_encoding = LabelEncoder()
        self.standard_scaler = StandardScaler()
        self.column_transformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])],
                                                    remainder='passthrough')

        self.load_dataset()

    def load_dataset(self):
        self.dataset = pd.read_csv('Churn_Modelling.csv')
        self.x = self.dataset.iloc[:, 3:-1].values
        self.y = self.dataset.iloc[:, -1].values

        self.label_encoding.fit(self.x[:, 2])
        self.x[:, 2] = self.label_encoding.transform(self.x[:, 2])
        self.column_transformer.fit(self.x)
        self.x = self.column_transformer.transform(self.x)
        self.standard_scaler.fit(self.x)
        self.x = self.standard_scaler.transform(self.x)

        self.x = np.array(self.x)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=0.8)

        x_tmp = np.array(self.x_test)
        y_tmp = np.array(self.y_test).reshape((len(self.y_test), 1))
        tmp_frame = pd.DataFrame(np.hstack((x_tmp, y_tmp)))
        tmp_frame.to_csv('Churn_Modelling_Test_Pre_Progress.csv')

    def compile_ann(self):
        self.ann = keras.models.Sequential()
        self.ann.add(keras.layers.Dense(units=6, activation='relu'))
        self.ann.add(keras.layers.Dense(units=6, activation='relu'))
        self.ann.add(keras.layers.Dense(units=1, activation='sigmoid'))
        # noinspection SpellCheckingInspection
        self.ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.ann.fit(self.x_train, self.y_train, validation_data=(self.x_test, self.y_test), batch_size=32, epochs=100)

    def pre_progressing(self, x: list[list]):
        x = np.array(x)
        x[:, 2] = self.label_encoding.transform(x[:, 2])
        x = self.column_transformer.transform(x)
        x = self.standard_scaler.transform(x)
        return np.array(x)

    def predict(self, x: list[list] | np.ndarray):
        return self.ann.predict(x)


basic_ann = BasicAnn()
basic_ann.compile_ann()
predict = basic_ann.predict(basic_ann.x_test)

print(f'{predict=}')

pre_progress = basic_ann.pre_progressing([[699, 'France', 'Female', 39, 1, 0, 2, 0, 0, 93826.63],
                                          [376, 'Germany', 'Female', 29, 4, 115046.74, 4, 1, 0, 119346.88]])
predict = basic_ann.predict(pre_progress)
print(f'{predict=}')
