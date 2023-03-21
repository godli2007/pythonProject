# #print
# print("hello world")

# #imput基础
# name = input("what is your name?")
# print("my name is", name)

# #raw用法
# print(r"\(~_~)/ \(~_~)/")

# #append基础
# L = ['Adam', 'Lisa', 'Bart']
# L.append('abc')
# print (L[3])

# #append扩展
# a=[]
# for i in range(5):
#   a.append([])
#   for j in range(5):
#     a[i].append(i)
# print (a)

# #insert基础
# L = ['Adam', 'Lisa', 'Bart']
# L.insert(0, 'Paul')
# print(L)

# #pop基础
# L = ['Adam', 'Lisa', 'Bart']
# L.pop()
# print(L)
# L.pop(0)
# print(L)

# #tuple元组
# t = ('Adam', 'Lisa', 'Bart')
# print(t)

# #tuple元组扩展
# t = ('a', 'b', ['A', 'B'])
# L = t[2]
# L[0] = 'X'
# L[1] = 'Y'
# print(t)

# #if基础
# age = 20
# if age >= 18:
#     print('your age is', age)
#     print ('adult')
# else:
#     print ('age is smaller than 18')
# print ('END')

# #if-elif-else基础
# age = int(input("年龄"))
# if age >= 18:
#     print('adult')
# elif age >= 6:
#     print('teenager')
# elif age >= 3:
#     print('kid')
# else:
#     print('baby')

# #for基础
# L = ['Adam', 'Lisa', 'Bart']
# for name in L:
#     print(name)

# #while基础
# N = 10
# x = 0
# while x<N:
#     print(x)
#     x = x+1

# #break基础
# sum = 0
# x = 1
# while True:
#     sum = sum + x
#     x = x + 1
#     if x > 100:
#         break
# print(sum)

# #continue基础
# L = [75, 98, 59, 81, 66, 43, 69, 85]
# M = 0
# sum = 0
# for x in L:
#     if x < 60:
#         continue
#     sum = sum + x
#     M = M +1
# print(sum/M)

# #多重循环
# for x in ['A', 'B', 'C']:
#     for y in ['1', '2', '3']:
#         print(x + y)
#
# #dic基础
# d = {'Adam': 95, 'Lisa': 85, 'Bart': 59, 'Paul': 72}
# print(d)
# if 'Paul' in d:
#     print(d['Paul'])
# print(d.get('Lisa'))
# print(d.get('abc'))

# #遍历dic基础
# d = {'Adam': 95, 'Lisa': 85, 'Bart': 59}
# for key in d:
#     print(key, d[key])

# #set基础
# s = set([1, 2, 3])
# s.add(0)  # set添加元素
# print(s)
# print('B' in s)
# s.remove(2)  # 删除set中的已有元素，删之前需判断元素是否存在，不存在的话会报错
# print(s)

# #单返回参数函数
# def my_abs(x):
#     if x >= 0:
#         return x
#     else:
#         return -x
#
#
# a = input('输入数字后求绝对值')
# print(my_abs(float(a)))

# #多重函数
# import math
#
#
# def move(x, y, step, angle):
#     nx = x + step * math.cos(angle)
#     ny = y - step * math.sin(angle)
#     return nx, ny
#
#
# x, y = move(100, 100, 60, math.pi / 6)
# print(x, y)
#
# r = move(100, 100, 60, math.pi / 6)
# print(r)

# #递归函数，阶乘
# def fact(n):
#     if n == 1:
#         return 1
#     return n * fact(n - 1)
#
#
# print(fact(3))
# print(fact(4))
# print(fact(100))


import pandas as pd


# 创建节假日数据框
holidays = pd.DataFrame({
  'holiday': ['春节', '清明', '五一', '端午', '中秋', '国庆'],
  'ds': pd.to_datetime(['2021-02-12', '2021-04-04', '2021-05-01', '2021-06-14', '2021-09-21', '2021-10-01',
                        '2022-02-01', '2022-04-05', '2022-05-01', '2022-06-03', '2022-09-10', '2022-10-01']),
})

print(holidays)