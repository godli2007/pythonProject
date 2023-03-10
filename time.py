# 导入必要的库
import pandas as pd
import numpy as np
from prophet import Prophet

# 读取数据集，假设是一个csv文件，有两列：ds和y
df = pd.read_csv('D:/data/inventory/inventory2020.csv')

# 创建Prophet模型对象
model = Prophet()

# 拟合模型到数据集
model.fit(df)

# 创建未来7天的日期框架
future = model.make_future_dataframe(periods=7)

# 预测未来7天的需求
forecast = model.predict(future)

# 打印预测结果
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])