# -*- coding:utf-8 -*-
# 导入模块
import pandas as pd
from prophet import Prophet

# 读取历史数据
df = pd.read_csv('D:/data/inventory/inventory2020.csv')

# 转换数据格式
df['ds'] = pd.to_datetime(df['date'])
df['y'] = df['demand']
df = df[['ds', 'y']]

# 创建并训练模型
model = Prophet()
model.fit(df)

# 创建未来日期
future = model.make_future_dataframe(periods=7)

# 生成预测结果
forecast = model.predict(future)

# 查看预测结果
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)