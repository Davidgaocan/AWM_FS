
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler



def get_scaler_data(data_path):
    df = pd.read_csv(data_path, header=None)
    X = df.iloc[:, :-1]

    # 创建一个标准化缩放器
    scaler = StandardScaler()

    # 对数据进行标准化
    X = scaler.fit_transform(X)

    # 将缩放后的数据重新转换为DataFrame
    X = pd.DataFrame(X)
    Y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test

def testalldata(dir):
    # 获取数据
    X_train, X_test, y_train, y_test = get_scaler_data(dir)
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(X_train , y_train.values.ravel())
    y_pred = clf.predict(X_test )
    acc_weighted = accuracy_score(y_test, y_pred)
    print(acc_weighted)

if __name__ == '__main__':
    testalldata('../data/uci_data_w/newzoo.data')



