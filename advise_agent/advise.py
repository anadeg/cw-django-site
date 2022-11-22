import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


DATA_FILE = "cw-data-2.csv"
TRAIN_FILE = "cw-train-2.csv"


class Advise:
    def __init__(self):
        self.X = None
        self.Y = None
        self.le = None

    def convert(self, file_train):
        data = pd.read_csv(file_train)

        data.drop(columns=['he_short'], inplace=True)   # for plots

        mapping = {"Брестская": 1, "Бресткая": 1, "Витебская": 2, "Гомельская": 3, "Гродненская": 4,
                   "Минск": 5, "Минская": 6, "Могилевская": 7, "Могилёвская": 7}
        data['region'].replace(mapping, inplace=True)

        data.fillna(0, inplace=True)

        self.le = LabelEncoder()
        data['he'] = self.le.fit_transform(data['he'])

        self.X = data.drop(['he'], axis=1)
        self.Y = data['he']

        self.X = self.features_engineering(self.X)

    @staticmethod
    def features_engineering(x):
        x['journ-int_com-theo'] = x['journ'] + x['int_com'] + x['theo']
        x['lingo-teach-psy'] = x['lingo'] + x['teach'] + x['psy']

        x['it-math'] = x['it'] + x['math']
        x['it-design'] = x['it'] + x['design']
        x['math-physics'] = x['math'] + x['physics']
        x['phys_cult-tourism'] = x['phys_cult'] + x['turism']
        x['ss-law'] = x['ss'] + x['law']
        x['ss-psy'] = x['ss'] + x['psy']
        x['ss-journ'] = x['ss'] + x['journ']
        x['ss-lingo-hist'] = x['ss'] + x['lingo'] + x['hist']
        x['phys_cult-bio'] = x['phys_cult'] + x['bio']
        x['chem-forest'] = x['chem'] + x['forest']

        x.drop(columns=['journ', 'int_com', 'theo',
                        'lingo', 'teach', 'psy',
                        'it', 'math',
                        'design',
                        'physics',
                        'phys_cult', 'turism',
                        'ss', 'law',
                        'hist',
                        'bio',
                        'chem', 'forest'],
               inplace=True)

        x = MinMaxScaler().fit_transform(np.array(x))
        return x

    def teach(self, models):
        self.model = self.__find_best(models)
        self.model.fit(self.X, self.Y)

    def advise(self, input):
        prediction = self.model.predict(input)
        prediction = self.le.inverse_transform(prediction)
        return prediction

    def __find_best(self, models):
        best_model = models[0]
        best = 0
        for model in models:
            results = cross_val_score(model, self.X, self.Y, cv=3)
            if best < results.mean():
                best = results.mean()
                best_model = model
        return best_model


if __name__ == '__main__':
    adv = Advise()
    adv.convert(TRAIN_FILE)

    lr = LogisticRegression()
    rfc = RandomForestClassifier()
    svc = SVC()
    knc = KNeighborsClassifier()
    xgb = XGBClassifier()

    models = [lr, rfc, svc, knc, xgb]
    adv.teach(models)

    predict = pd.read_csv('cw-predict.csv')
    mapping = {"Брестская": 1, "Бресткая": 1, "Витебская": 2, "Гомельская": 3, "Гродненская": 4,
               "Минск": 5, "Минская": 6, "Могилевская": 7, "Могилёвская": 7}
    predict['region'].replace(mapping, inplace=True)
    predict.fillna(0, inplace=True)
    predict = adv.features_engineering(predict)

    y_true = adv.Y
    y_pred = adv.model.predict(adv.X)
    print('Accuracy score --- ', accuracy_score(y_true, y_pred))

    predict = np.array(predict)
    results = adv.advise(predict)
    for i, res in enumerate(results):
        print(f'{i + 1} --- {res}')
