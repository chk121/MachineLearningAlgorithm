import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def build_gbdt(x, y, consame=5, maxiter=10000, shrink=0.0005):
    """
    建立函数构建GBDT模型
    x: sepal length萼片长度, sepal width萼片宽度, petal length花瓣长度
    y: Petal.Width
    consame: 连续consame次 残差平方和相等时，迭代终止
    
    """
    f0 = np.mean(y)
    
    rss = []
    model_list = [f0]

    for i in tqdm(range(maxiter), ncols=80):
        ## 损失函数是平方损失函数时，负梯度即为残差
        residual = y - f0
        ## 根据残差学习一颗回归树，分割点满足的最小样本量为30
        clf = tree.DecisionTreeRegressor(min_samples_leaf=30)
        clf = clf.fit(x, residual)
        ## 更新回归树，并生成新的估计结果
        model_list.append(clf)
        f0 += shrink * clf.predict(x)
        ## 统计残差平方和
        rss.append(np.sum((f0-y)**2))

        if len(rss) >= consame and np.std(rss[len(rss)-consame:]) == 0:
            print("共迭代{}次，满足终止条件！迭代退出".format(i+1))
            break

    return rss, model_list


def gbdt_predict(x, model_ls, shrink):
    f0 = model_ls[0]
    for i in range(1, len(model_ls)):
        f0 += shrink * model_ls[i].predict(x)

    return f0
    


if __name__ == '__main__':
    url = 'https://www.gairuo.com/file/data/dataset/iris.data'
    iris = pd.read_csv(url)
    x, y = iris.drop(columns=['species', 'petal_width']), iris['petal_width']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=1)
    # print(x_test)
    ## rss残差平方和(residual square sum)
    rss, model_list = build_gbdt(x_train, y_train)
    # print(pd.Series(rss).describe())
    print('GBDT误差平方和: {}'.format(np.sum((y_test - gbdt_predict(x_test, model_list, .0005))**2)))
    

    clf = tree.DecisionTreeRegressor(min_samples_leaf=30)
    clf = clf.fit(x_train, y_train)
    print('回归决策树误差平方和: {}'.format(np.sum((y_test - clf.predict(x_test))**2)))

    """
    ## rss随iter次数的变化曲线图
    sv_path = 'test.pdf'
    with PdfPages(sv_path) as pdf:

        plt.plot(range(10000), rss, '-', c='black', linewidth=3)
        plt.xlabel("iter", fontsize=13)
        plt.ylabel("RSS", fontsize=12)
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
    """
