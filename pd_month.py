# -*- coding:utf-8 -*-
# 导入模块
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error

# 读取历史数据
df = pd.read_csv('D:/data/inventory/inventory2020-2022.csv')


# 按月汇总数据
df = df.groupby('date').sum().reset_index()
# 转换数据格式
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['demand']
df = df[['ds', 'y']]
# 重采样数据为月频率
df = df.resample('M', on='ds').sum().reset_index()

# 可以修改季节性参数（seasonality_mode, seasonality_prior_scale等），趋势参数（changepoint_prior_scale, n_changepoints等），节假日参数（holidays, holidays_prior_scale等）
# 创建并训练模型
model = Prophet(seasonality_mode='multiplicative', n_changepoints=50)
model.fit(df)

# 创建未来日期
future = model.make_future_dataframe(periods=3, freq='M')

# 生成预测结果
forecast = model.predict(future)

# 查看预测结果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(3))



# # 绘制预测图表
# fig1 = px.line(forecast, x='ds', y=['yhat', 'yhat_lower', 'yhat_upper'])
# fig1.show()
# fig2 = px.bar(forecast, x='ds', y=['trend', 'seasonal', 'holiday'])
# fig2.show()


# 绘制预测图表
fig1 = model.plot(forecast)
# fig1.show()
fig2 = model.plot_components(forecast)
# fig2.show()
# plt.show(block=False)
plt.show()

# print('The mean absolute error is:', mean_absolute_error(df['y'], forecast['yhat']))

# 进行交叉验证
df_cv = cross_validation(model, initial='365.25  days', period='30 days', horizon='90 days')

# 计算各种指标
df_p = performance_metrics(df_cv)


# 输出MAE, MAPE, RMSE
print('MAE:', df_p['mae'].mean())
# print('MAPE:', df_p['mape'].mean())
print('RMSE:', df_p['rmse'].mean())

# # 绘制误差分布图
# fig3 = px.histogram(df_cv, x='yhat - y')
# fig3.show()
# # 绘制误差随时间变化图
# fig4 = px.scatter(df_cv, x='ds', y='yhat - y')
# fig4.show()