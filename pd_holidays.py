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
df = pd.read_csv('D:/data/inventory/L220_inventory2020-2022.csv')

# 转换数据格式
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['demand']
df = df[['ds', 'y']]

# 创建并训练模型
# changepoint_prior_scale: 这个参数控制了趋势变化点的灵敏度，值越大，拟合的跟随性越好，可能会过拟合

model = Prophet(seasonality_mode='multiplicative', n_changepoints=50, holidays=holidays)
model.fit(df)
# , holidays=holidays

# 创建未来日期
future = model.make_future_dataframe(periods=7)

# 生成预测结果
forecast = model.predict(future)

# 查看预测结果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))

# ds：时间序列的时间戳
# yhat：时间序列的预测值
# yhat_lower：预测值的下界
# yhat_upper：预测值的上界

# 绘制预测图表
fig1 = model.plot(forecast)
fig2 = model.plot_components(forecast)
plt.show()

# 进行交叉验证
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
fig3 = plot_cross_validation_metric(df_cv, metric='mse')
fig4 = plot_cross_validation_metric(df_cv, metric='rmse')
fig5 = plot_cross_validation_metric(df_cv, metric='mae')
fig6 = plot_cross_validation_metric(df_cv, metric='mdape')
fig7 = plot_cross_validation_metric(df_cv, metric='smape')
fig8 = plot_cross_validation_metric(df_cv, metric='coverage')
plt.show()