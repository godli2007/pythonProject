# -*- coding:utf-8 -*-
# 导入模块
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px
from prophet.diagnostics import cross_validation, performance_metrics
# 导入可视化函数
from prophet.plot import plot_cross_validation_metric

# 创建自定义节假日数据框
holidays = pd.DataFrame({
    'holiday': ['春节', '清明节', '劳动节', '端午节', '中秋节', '国庆节'],
    'ds': pd.to_datetime(['2020-01-25', '2020-04-04', '2020-05-01', '2020-06-25',
                          '2020-10-01', '2020-10-01',
                          # 其他年份的日期
                          ]),
    # 增加持续时间（以天为单位）
    # 如果不指定，默认为1天
    # 如果指定为负数，表示该日期之前的天数也是节假日
    # 如果指定为正数，表示该日期之后的天数也是节假日
    # 例如，春节通常持续7天，所以可以指定为6
    # 国庆节通常持续3天，所以可以指定为2
    # 清明节通常只有一天，所以可以不指定或者指定为0
    'lower_window': [-1, -1, -1, -1, -1, -1],
    'upper_window': [6, 0, 2, 2, 2, 2],
    # 增加国家/地区（可选）
    # 如果不指定，默认为所有国家/地区都适用该节假日效应
    # 如果指定了国家/地区，那么只有当训练数据或预测数据中包含了相应的国家/地区列时，
    # 才会考虑该节假日效应
    # 国家/地区的名称必须与ISO标准一致（两位字母代码）

    # 'country': ['CN','CN','CN','CN','CN','CN']
})

# 读取历史数据
df = pd.read_csv('L220_inventory2020-2022.csv')

# 转换数据格式
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['demand']
df = df[['ds', 'y']]

# 创建并训练模型
# seasonality_mode: 季节模型方式，'additive'(加法模型) (默认) 或者 'multiplicative'（乘法模型）
# 加法季节性意味着季节性的影响是和趋势相加得到预测值，这适用于季节性变化相对稳定的情况。乘法季节性意味着季节性的影响是和趋势相乘得到预测值，这适用于季节性变化随着趋势增长而增大的情况。

model = Prophet(seasonality_mode='multiplicative', n_changepoints=50, holidays=holidays)
model.fit(df)

# 创建未来日期
future = model.make_future_dataframe(periods=30)
# future['floor'] = 0

# 生成预测结果
forecast = model.predict(future)

# 查看预测结果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

# ds：时间序列的时间戳
# yhat：时间序列的预测值
# yhat_lower：预测值的下界
# yhat_upper：预测值的上界

# 绘制预测图表
fig = model.plot(forecast)
# 观测值demand：时间序列的历史数据，用黑色点表示。
# 预测值yhat：时间序列的未来数据，用蓝色线表示。
# 置信区间yhat_lower\yhat_upper：预测值的不确定性范围，用蓝色阴影区域表示
fig = model.plot_components(forecast)
# 趋势trends：反映了时间序列的长期变化，可以是线性、对数或平坦的。
# 节假日holidays：反映了特定日期或时间段对时间序列的影响，可以是固定或可变的。
# 周期性yearly\weekly：反映了时间序列在不同时间尺度上的重复模式，例如周、月、年等。
plt.show()

# 进行交叉验证
# model: model是已经训练的Prophet模型
# horizon: horizon是每次预测所使用的数据的时间长度，比如‘30d’（30天）
# period：period是触发预测动作的时间周期。如果设置为‘7d’，01-07、01-14、01-21等等，而这些预测的数据为前面定义的horizon。这个值的默认值为horizon*0.5
# Initial：整个交叉验证的数据范围，结束点是昨天的点，开始点是（昨天-initial)，initial的默认值是3*horizon。当然同学们也可根据实际情况手动设置，比如“110d”。
df_cv = cross_validation(model, initial='365.25 days', period='30 days', horizon='90 days')

# 查看交叉验证结果
print(df_cv.head())

# cutoff：用于评估预测性能的滚动窗口
# y：时间序列的取值

# 计算性能指标
df_p = performance_metrics(df_cv)

# 查看性能指标结果
print(df_p.head())

# 绘制预测值和真实值的对比图
fig = plot_cross_validation_metric(df_cv, metric='mse')
fig = plot_cross_validation_metric(df_cv, metric='rmse')
fig = plot_cross_validation_metric(df_cv, metric='mae')
fig = plot_cross_validation_metric(df_cv, metric='mdape')
fig = plot_cross_validation_metric(df_cv, metric='smape')
fig = plot_cross_validation_metric(df_cv, metric='coverage')
# plt.show()

# mse：均方误差，即真实值和预测值之差平方后求平均，反映了误差大小和方向。
# rmse：均方根误差，即mse开平方根，反映了误差大小和方向，并且与原始数据单位一致。
# mae：平均绝对误差，即真实值和预测值之差绝对值后求平均，反映了误差大小而不考虑方向。
# mdape：中位数绝对百分比误差，即真实值和预测值之比减去一再取绝对值后求中位数，反映了相对误差水平而不考虑方向。
# smape：对称平均绝对百分比误差，即真实值和预测值之差除以它们两者平均后取绝对值再求平均，反映了相对误差水平而不考虑方向，并且避免了零除问题。
# coverage：置信区间覆盖率，即真实值落在置信区间内的比例，反映了置信区间是否合理地捕捉了真实变化范围。

# 通常来说，在选择评价指标时要考虑你关心什么样的错误特征。例如如果你想要惩罚大错误而不是小错误，则可以选择mse或rmse；如果你想要忽略错误符号而只关心错误幅度，则可以选择mae或mdape；如果你想要把错误归一化到原始数据范围，则可以选择smape等等。
