# -*- coding:utf-8 -*-

# 导入相关库
import pandas as pd
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
import numpy as np

# 创建节假日数据框
holidays = pd.DataFrame({
      'holiday': ['春节', '清明节', '劳动节', '端午节', '中秋节', '国庆节'],
    'ds': pd.to_datetime(['2020-01-25', '2020-04-04', '2020-05-01', '2020-06-25',
                          '2020-10-01', '2020-10-01',
                           # 其他年份的日期
                          ]),
})

# 读取需求数据
df = pd.read_csv('D:/data/inventory/inventory2020-2022.csv')

# 转换数据格式
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['demand']
df = df[['ds', 'y']]

# 定义评价指标（如平均绝对百分比误差MAPE）
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 定义参数网格
param_grid = {
    "changepoint_prior_scale" : [0.001, 0.01, 0.1, 0.5],
    "seasonality_mode" : ["additive", "multiplicative"],
    "growth": ["linear", "logistic"]
}

# 初始化最优参数和最优误差
best_params = None
best_mape = float("inf")

# 遍历参数网格
for params in ParameterGrid(param_grid):

    # 创建prophet模型对象，并传入节假日数据框和当前参数
    model = Prophet(holidays=holidays, **params)

    # 训练模型
    model.fit(df)


    # 创建未来日期数据框（假设预测未来30天）
    future = model.make_future_dataframe(periods=30)

    # 生成预测结果
    forecast = model.predict(future)

    # 计算MAPE误差（只计算最后30天）
    mape_error = mape(df['y'][-30:], forecast['yhat'][-30:])

    # 如果当前误差小于最优误差，则更新最优参数和最优误差
    if mape_error < best_mape:
        best_params = params
        best_mape = mape_error

# 打印最优参数和最优误差
print(best_params)
print(best_mape)

model.plot(forecast)

model.plot_components(forecast)
