# -*- coding:utf-8 -*-
# 导入模块
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import plotly.express as px
from prophet.diagnostics import cross_validation, performance_metrics

# 读取历史数据
df = pd.read_csv('D:/data/inventory/inventory2020-2022.csv')


# 转换数据格式
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['demand']
df = df[['ds', 'y']]

# holidays = pd.DataFrame({
#   'holiday': ['春节', '清明节', '劳动节', '端午节', '中秋节', '国庆节'],
#   'ds': pd.to_datetime(['2020-01-24', '2020-04-04', '2020-05-01', '2020-06-03', '2020-09-21', '2020-10-01']),
# })

# 创建并训练模型
model = Prophet(seasonality_mode='multiplicative', n_changepoints=50)
model.fit(df)
# , holidays=holidays

# 创建未来日期
future = model.make_future_dataframe(periods=7)

# 生成预测结果
forecast = model.predict(future)

# 查看预测结果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))



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
plt.show()

# 进行交叉验证
df_cv = cross_validation(model, initial='365.25 days', period='30 days', horizon='90 days')

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