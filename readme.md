```
          ______ _____ 
    /\   |  ____|_   _|
   /  \  | |__    | |  
  / /\ \ |  __|   | |  
 / ____ \| |____ _| |_ 
/_/    \_\______|_____|

```
使用自编码器进行缺失插补

Griswold1999.csv不含有多态
tnt建树
1. settings->memory->max tree 修改保存的最大树的数量
2. analyze->suboptimal 修改距最佳得分的树的保留空间

实验流程设计：
1. 在pycharm中使用缺失插补方法，对缺失数据集进行处理
2. 在tnt中建树，比较不同建树结果与论文树的RF，co distance距离
3. 在R中使用treespace，画出不同树的MSD散点图，比较不同建树结果与论文树的treeVec,nNodes距离。