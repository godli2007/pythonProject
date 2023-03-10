# -*- coding:utf-8 -*-
# 导入模块
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 读取历史数据
df = pd.read_csv('D:/data/inventory/inventory2020.csv')


# 转换数据格式
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['demand']
df = df[['ds', 'y']]

holidays = pd.DataFrame({
  'holiday': ['春节', '清明节', '劳动节', '端午节', '中秋节', '国庆节'],
  'ds': pd.to_datetime(['2020-01-24', '2020-04-04', '2020-05-01', '2020-06-03', '2020-09-21', '2020-10-01']),
})

# 创建并训练模型
model = Prophet(seasonality_mode='multiplicative', n_changepoints=50, holidays=holidays)
model.fit(df)


# 创建未来日期
future = model.make_future_dataframe(periods=30)

# 生成预测结果
forecast = model.predict(future)

# 查看预测结果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30))

# 绘制预测图表
model.plot(forecast)
model.plot_components(forecast)
plt.show()
