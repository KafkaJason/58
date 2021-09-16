## 当前项目目录如下 
    .
    ├── README.md
    ├── data
    |   ├── train.txt (训练集)
    |   └── test.txt (测试集)
    └── code
        ├── feature (运行中途生成文件减少内存占用)
        ├── result (单模型结果保存)
        ├── base1.py (模型文件1)
        └── base1.py (模型文件2)

## 代码执行顺序
由于在比赛中使用了两套方案进行融合，两套方案特征方面差别不大，主要在模型部分有比较大的差异。读者可以只选择一个运行。方案有很多不足的地方，有欢迎各位大佬一起探讨。
依次执行 base1.py, base2.py

## 结果文件

结果文件: ./code/final_result.csv

## 解决方案
http://kafkajason.top/index.php/archives/9/

## 数据下载网址
https://tech.58.com/game/problemDesc?contestId=4&token=58tech
