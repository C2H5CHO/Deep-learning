# 一、简单线性回归的机器学习建模思路
&emsp;&emsp;在机器学习的众多模型中，线性回归是最基础也最易理解的入门模型。它的核心目标是找到一条能最好地**贴合**数据点的直线（或高维空间中的超平面），而这个**最好**的标准，正是通过优化思想来定义和实现的。理解线性回归的建模逻辑，能帮助我们掌握机器学习中**通过数据寻找最优参数**的通用思路。
## 1.1 建模问题
简单线性回归的本质是用一元线性方程描述自变量与因变量之间的关系。假设我们有一组二维数据点，每个点都可以表示为$(x_i, y_i)$，其中$x_i$是自变量，$y_i$是因变量。我们希望找到一个线性方程$y = ax + b$，使得对于每个$x_i$，方程的预测值$\hat{y}_i = ax_i + b$尽可能接近真实值$y_i$。

为什么要选择线性方程来拟合？因为线性关系是自然界中最常见的关系之一，且线性模型结构简单、解释性强。比如，在房价预测中，面积（自变量）与房价（因变量）可能呈线性关系；在销量预测中，广告投入与销量也可能近似线性相关。即使数据存在非线性关系，线性模型也常作为基础模型，为更复杂的模型（如多项式回归、神经网络）提供参考。

```python
import torch
import matplotlib.pyplot as plt

# 定义数据点：x坐标为[1,3]，y坐标为[2,4]
A = torch.arange(1, 5).reshape(2, 2).float()

# 绘制散点图
plt.figure(figsize=(6, 8))
plt.plot(A[:, 0], A[:, 1], 'o', color='blue', markersize=8)  # A[:,0]取x坐标，A[:,1]取y坐标
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
```
- 运行结果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/040b9571061b4f11bc033b7c179f4cec.png#pic_center)
- 示例解读：
> 假设，我们有两个数据点：(1, 2)和(3, 4)。从直观上看，这两个点分布在一条直线附近，我们的任务就是找到这条直线的斜率$a$和截距$b$。而从图中能清晰看到它们的位置关系——(1, 2)在左下方，(3, 4)在右上方，呈现明显的正相关趋势。

## 1.2 优化问题
要找到**最好**的直线$y = ax + b$，首先需要明确**好**的标准。这个标准需要通过数学方式量化，也就是我们常说的**损失函数**；而寻找使损失函数最小的参数$a$和$b$的过程，就是**优化问题**。
### 1.2.1 定义误差指标
对于每个数据点$(x_i, y_i)$，预测值$\hat{y}_i$与真实值$y_i$的差距称为**误差**，即$e_i = y_i - \hat{y}_i$。在上述*1.1*的例子中：

- 对于点(1, 2)，误差为$e_1 = 2 - (a \times 1 + b) = 2 - a - b$
- 对于点(3, 4)，误差为$e_2 = 4 - (a \times 3 + b) = 4 - 3a - b$

误差可能为正（预测值偏小）也可能为负（预测值偏大）。如果直接将误差相加，正负误差可能相互抵消，无法真实反映总误差。比如，若$e_1 = 2$、$e_2 = -2$，总和为0，但实际每个点的误差都很大。
### 1.2.2 构造损失函数
为了避免正负抵消，我们通常用**误差的平方**来衡量单个点的误差，再将所有点的平方误差相加，得到**误差平方和（SSE）**：

$$SSE = e_1^2 + e_2^2 = (2 - a - b)^2 + (4 - 3a - b)^2$$

SSE的值越小，说明预测值与真实值的整体差距越小，直线的拟合效果越好。因此，线性回归问题就转化为：寻找$a$和$b$的取值，使SSE达到最小值。这里的SSE就是**损失函数**——它是关于参数$a$和$b$的函数，我们的优化目标就是最小化这个函数。

为什么选择平方误差而非绝对值误差？除了避免正负抵消，平方误差还有一个重要特性：它对较大的误差更敏感（平方会放大误差），这能迫使模型更关注偏差大的点，从而提高整体拟合精度。此外，平方函数是光滑可导的，便于后续用数学方法求解最小值，而绝对值函数在0点不可导，会增加求解难度。
## 1.3 最优化问题的求解
要找到SSE的最小值，需要结合函数的性质选择合适的数学工具。从上述*1.1*的可视化结果可知，SSE是一个**凸函数**，而凸函数的最小值点有明确的数学特征——导数（或偏导数）为0的点。
### 1.3.1 凸函数的性质
凸函数是指函数图像整体**向上凸**的函数，它有且仅有一个最低点（全域最小值）。用*1.1*的$y = x^2$举例：对于任意两个点$x_1$和$x_2$，两点连线的中点的函数值，始终大于等于两点函数值的平均值，即$\frac{f(x_1) + f(x_2)}{2} \geq f(\frac{x_1 + x_2}{2})$。

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置字体，支持中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 定义a和b的取值范围
a_range = np.arange(-1, 3, 0.05)  # a从-1到3，步长0.05
b_range = np.arange(-1, 3, 0.05)  # b从-1到3，步长0.05

# 生成网格点
a_grid, b_grid = np.meshgrid(a_range, b_range)

# 计算每个(a,b)对应的SSE
SSE = (2 - a_grid - b_grid)**2 + (4 - 3*a_grid - b_grid)** 2

# 绘制3D曲面和等高线
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 3D曲面
surf = ax.plot_surface(a_grid, b_grid, SSE, cmap='rainbow', alpha=0.8)
# 等高线（投影到z=0平面）
ax.contour(a_grid, b_grid, SSE, zdir='z', offset=0, cmap='rainbow')

ax.set_xlabel('a（斜率）')
ax.set_ylabel('b（截距）')
ax.set_zlabel('SSE（误差平方和）')
ax.set_title('SSE的3D可视化')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
```
- 运行结果：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/715729bd69fb4a8e9597f51da959db0a.png#pic_center)
- 示例解读：
> SSE作为误差平方和，也是一个凸函数。从3D可视化图像可以看到，SSE的图像像一个**碗**，最低点就是我们要找的最优参数点。对于凸函数，只要找到导数为0的点，就能确定这个点就是最小值点，无需担心局部最小值的问题。
### 1.3.2 用偏导数求解最小值
SSE是关于$a$和$b$的二元函数，要找到其最小值，需分别对$a$和$b$求偏导数，并令偏导数为0，联立方程求解。

#### （1）Step 1：求SSE对$a$的偏导数
根据链式法则，对于函数$f(u) = u^2$，其导数为$f'(u) = 2u$。因此：
$$
\frac{\partial SSE}{\partial a} = 2(2 - a - b) \cdot \frac{\partial (2 - a - b)}{\partial a} + 2(4 - 3a - b) \cdot \frac{\partial (4 - 3a - b)}{\partial a}
$$
其中，$\frac{\partial (2 - a - b)}{\partial a} = -1$，$\frac{\partial (4 - 3a - b)}{\partial a} = -3$，代入得：
$$
\frac{\partial SSE}{\partial a} = 2(2 - a - b)(-1) + 2(4 - 3a - b)(-3)
$$
展开化简：
$$
\frac{\partial SSE}{\partial a} = -2(2 - a - b) - 6(4 - 3a - b) = -4 + 2a + 2b - 24 + 18a + 6b = 20a + 8b - 28
$$
令偏导数为0，得到方程 1：$20a + 8b - 28 = 0$。

#### （2）Step 2：求SSE对$b$的偏导数
同理，$\frac{\partial (2 - a - b)}{\partial b} = -1$，$\frac{\partial (4 - 3a - b)}{\partial b} = -1$，代入得：
$$
\frac{\partial SSE}{\partial b} = 2(2 - a - b)(-1) + 2(4 - 3a - b)(-1)
$$
展开化简：
$$
\frac{\partial SSE}{\partial b} = -2(2 - a - b) - 2(4 - 3a - b) = -4 + 2a + 2b - 8 + 6a + 2b = 8a + 4b - 12
$$
令偏导数为0，得到方程 2：$8a + 4b - 12 = 0$。

#### （3）Step 3：联立方程求解
用*方程 1*减去*方程 2* 的2倍，消去$b$：
$$(20a + 8b - 28) - 2 \times (8a + 4b - 12) = 0$$

$$20a + 8b - 28 - 16a - 8b + 24 = 0$$

$$4a - 4 = 0 \implies a = 1$$
将$a = 1$代入*方程 2*：
$$
8 \times 1 + 4b - 12 = 0 \implies 4b = 4 \implies b = 1
$$

```python
import torch

# 定义可微分的参数a和b（初始值设为1）
a = torch.tensor(1.0, requires_grad=True)  # requires_grad=True表示需要计算梯度
b = torch.tensor(1.0, requires_grad=True)

# 计算SSE
sse = (2 - a - b)**2 + (4 - 3*a - b)** 2

# 计算偏导数
grads = torch.autograd.grad(sse, [a, b])
print(f"对a的偏导数：{grads[0]}")
print(f"对b的偏导数：{grads[1]}")
```
- 运行结果：
```
对a的偏导数：-0.0
对b的偏导数：-0.0
```
- 示例解读：
> 当$a = 1$、$b = 1$时，SSE取得最小值，对应的直线为$y = x + 1$。这个结果也可以通过PyTorch的自动微分工具验证：当$a$和$b$都为1时，SSE对两者的偏导数均为0，证明这是最小值点。

## 1.4 机器学习的建模流程
从简单线性回归的例子中，我们可以提炼出机器学习建模的通用流程，这一流程同样适用于复杂的深度学习模型：
### 1.4.1 Step 1：提出基本模型
根据问题的性质选择合适的模型结构。模型是对数据关系的假设，比如线性回归用直线假设数据的线性关系，神经网络用多层非线性变换假设复杂关系。模型中包含需要通过数据求解的参数（如线性回归中的$a$和$b$，神经网络中的权重和偏置）。
### 1.4.2 Step 2：确定损失函数
损失函数是衡量模型预测效果的**标尺**，它量化了预测值与真实值的差距。损失函数的设计需结合问题类型：回归问题常用SSE、MAE（平均绝对误差）；分类问题常用交叉熵损失。损失函数必须是关于模型参数的函数，这样才能通过调整参数降低损失。
### 1.4.3 Step 3：选择优化方法求解
根据损失函数的性质（是否为凸函数、是否可导等）选择优化方法。对于凸函数（如线性回归的SSE），可直接用最小二乘法求解；对于非凸函数（如神经网络的损失函数），常用梯度下降等迭代方法。优化的目标是找到使损失函数最小的参数，此时的模型就是**训练好的模型**。

这三个步骤环环相扣：模型决定了参数的形式，损失函数定义了**好**的标准，优化方法则是找到最优参数的工具。理解这一流程，是掌握机器学习建模的核心。
# 二、最小二乘法
&emsp;&emsp;最小二乘法是求解线性回归问题的经典优化算法，它通过严格的数学推导直接得到最优参数，无需迭代计算。这种**直接求解**的特性使其在小规模线性问题中高效且准确，同时也是理解更复杂优化算法的基础。
## 2.1 最小二乘法的代数表示
对于更一般的简单线性回归场景（包含$m$个数据点），模型为$y = wx + b$，其中$w$是斜率（权重），$b$是截距。我们需要通过这$m$个点$(x_1,y_1), (x_2,y_2), ..., (x_m,y_m)$，求解使SSE最小的$w$和$b$。
### 2.1.1 损失函数的一般形式
SSE的表达式为：
$$
SSE = \sum_{i=1}^m (y_i - (wx_i + b))^2
$$
### 2.1.2 求解最优参数$w$和$b$
通过对$w$和$b$求偏导数并令其为0，可推导出通用公式（推导过程与两个点的例子类似，只是扩展到$m$个点）：
$$w = \frac{\sum_{i=1}^m y_i(x_i - \bar{x})}{\sum_{i=1}^m x_i^2 - \frac{1}{m}(\sum_{i=1}^m x_i)^2}$$

$$b = \bar{y} - w\bar{x}$$
其中，$\bar{x} = \frac{1}{m}\sum_{i=1}^m x_i$是$x$的均值，$\bar{y} = \frac{1}{m}\sum_{i=1}^m y_i$是$y$的均值。

这个公式的意义在于：$w$的分子衡量了$y$与$x$偏离各自均值的协变程度，分母衡量了$x$的离散程度（方差的$m$倍），因此$w$本质上反映了$x$和$y$的线性相关程度；而$b$则是当$x$取均值时，$y$的均值与$w$乘以$x$均值的差值，确保直线经过数据的**中心**附近。
### 2.1.3 实例验证
用*1.1*的两个点$(1,2)$和$(3,4)$验证：

- $m = 2$，$\bar{x} = \frac{1 + 3}{2} = 2$，$\bar{y} = \frac{2 + 4}{2} = 3$；
- 分子：$\sum y_i(x_i - \bar{x}) = 2 \times (1 - 2) + 4 \times (3 - 2) = 2 \times (-1) + 4 \times 1 = -2 + 4 = 2$；
- 分母：$\sum x_i^2 - \frac{1}{m}(\sum x_i)^2 = (1^2 + 3^2) - \frac{1}{2}(1 + 3)^2 = (1 + 9) - \frac{1}{2} \times 16 = 10 - 8 = 2$；
- 因此，$w = \frac{2}{2} = 1$，$b = 3 - 1 \times 2 = 1$，与之前的结果一致。

```python
import torch

# 数据点：x=[1,3]，y=[2,4]
x = torch.tensor([1.0, 3.0])
y = torch.tensor([2.0, 4.0])
m = len(x)  # 数据点数量

# 计算均值
x_mean = x.mean()
y_mean = y.mean()

# 计算w
numerator = torch.sum(y * (x - x_mean))  # 分子：sum(y_i*(x_i - x_mean))
denominator = torch.sum(x**2) - (torch.sum(x)** 2) / m  # 分母：sum(x_i²) - (sum(x_i))²/m
w = numerator / denominator

# 计算b
b = y_mean - w * x_mean

print(f"w = {w.item()}")
print(f"b = {b.item()}")
```
- 运行结果：
```
w = 1.0
b = 1.0
```
## 2.2 最小二乘法的矩阵表示
当数据量较大或特征维度较高时，用矩阵表示数据和参数能极大简化计算，同时更适合计算机实现（矩阵运算可高效并行）。
### 2.2.1 矩阵形式的定义
为了将截距$b$纳入参数向量，我们对模型和数据做如下扩展：

- 定义参数向量$\hat{w} = (w, b)$，包含斜率和截距；
- 定义特征向量$\hat{x}_i = (x_i, 1)$，在每个$x_i$后增加常数1，用于与截距$b$相乘；
- 特征矩阵$X$（形状为$m \times 2$）：每行是一个特征向量$\hat{x}_i$，即  
  $X = \begin{bmatrix} x_1 & 1 \\ x_2 & 1 \\ ... & ... \\ x_m & 1 \end{bmatrix}$；
- 标签向量$y$（形状为$m \times 1$）：$y = \begin{bmatrix} y_1 \\ y_2 \\ ... \\ y_m \end{bmatrix}$。

此时，模型可表示为$y = X \hat{w}^T$（$\hat{w}^T$是$\hat{w}$的转置）。
### 2.2.2 矩阵形式的损失函数与求解
SSE的矩阵表示为：
$$
SSE = \| y - X\hat{w}^T \|_2^2 = (y - X\hat{w}^T)^T (y - X\hat{w}^T)
$$
其中，$\| \cdot \|_2$是L2范数（向量元素平方和的开方）。

对$\hat{w}$求导并令其为0（利用矩阵求导法则），可推导出最优参数的矩阵公式：
$$
\hat{w}^T = (X^T X)^{-1} X^T y
$$
其中，$X^T$是$X$的转置，$(X^T X)^{-1}$是$X^T X$的逆矩阵。

这个公式的意义是：通过矩阵运算将特征与标签的关系**解耦**，直接得到最优参数。当$X^T X$可逆时（即数据不存在完全共线性），该公式有效。
### 2.2.3实例验证
假设，$X = \begin{bmatrix}1 & 1 \\ 3 & 1\end{bmatrix}$，$y = \begin{bmatrix}2 \\ 4\end{bmatrix}$：

- $X^T X = \begin{bmatrix}1 & 3 \\ 1 & 1\end{bmatrix} \begin{bmatrix}1 & 1 \\ 3 & 1\end{bmatrix} = \begin{bmatrix}1 \times 1 + 3 \times 3 & 1 \times 1 + 3 \times 1 \\ 1 \times 1 + 1 \times 3 & 1 \times 1 + 1 \times 1\end{bmatrix} = \begin{bmatrix}10 & 4 \\ 4 & 2\end{bmatrix}$；
- $(X^T X)^{-1} = \frac{1}{10 \times 2 - 4 \times 4} \begin{bmatrix}2 & -4 \\ -4 & 10\end{bmatrix} = \frac{1}{4} \begin{bmatrix}2 & -4 \\ -4 & 10\end{bmatrix} = \begin{bmatrix}0.5 & -1 \\ -1 & 2.5\end{bmatrix}$；
- $X^T y = \begin{bmatrix}1 & 3 \\ 1 & 1\end{bmatrix} \begin{bmatrix}2 \\ 4\end{bmatrix} = \begin{bmatrix}1 \times 2 + 3 \times 4 \\ 1 \times 2 + 1 \times 4\end{bmatrix} = \begin{bmatrix}14 \\ 6\end{bmatrix}$；
- $\hat{w}^T = (X^T X)^{-1} X^T y = \begin{bmatrix}0.5 & -1 \\ -1 & 2.5\end{bmatrix} \begin{bmatrix}14 \\ 6\end{bmatrix} = \begin{bmatrix}0.5 \times 14 - 1 \times 6 \\ -1 \times 14 + 2.5 \times 6\end{bmatrix} = \begin{bmatrix}1 \\ 1\end{bmatrix}$，与之前结果一致。

```python
import torch

# 1. pytorch矩阵求解
# 定义特征矩阵X和标签向量y
X = torch.tensor([[1.0, 1.0], [3.0, 1.0]])  # 每行是(x_i, 1)
y = torch.tensor([[2.0], [4.0]])  # 形状为(2,1)

# 计算X^T * X
X_T_X = torch.mm(X.T, X)  # mm表示矩阵乘法

# 计算(X^T * X)的逆矩阵
X_T_X_inv = torch.inverse(X_T_X)

# 计算最优参数w^T = (X^T X)^{-1} X^T y
w = torch.mm(torch.mm(X_T_X_inv, X.T), y)

print("最优参数：")
print(w)

print('--'*50)
# 2. lstsq 求解最小二乘法函数
result = torch.linalg.lstsq(X, y)
print("最优参数：")
print(result.solution)
```
- 运行结果：
```
最优参数：
tensor([[1.0000],
        [1.0000]])
----------------------------------------------------------------------------------------------------
最优参数：
tensor([[1.0000],
        [1.0000]])
```
## 2.3 最小二乘法的意义与局限
最小二乘法的核心意义在于：它为线性回归提供了一种**解析解**（通过公式直接求解），无需迭代，计算精准。这使其在小规模数据、低维度特征的场景中非常高效，也是统计学中分析线性关系的基础工具。

但它也有局限：当特征维度极高（如超过10万）时，计算$(X^T X)^{-1}$的时间和空间成本会急剧增加；当数据存在多重共线性（$X^T X$不可逆）时，公式无法使用，需要引入正则化（如岭回归）；此外，它仅适用于线性模型和SSE损失函数，对于非线性模型或其他损失函数（如MAE），则需要依赖梯度下降等迭代算法。

尽管如此，最小二乘法仍是理解优化思想的重要起点——它展示了如何通过数学推导将**拟合数据**转化为**求解方程**，这种思路也为后续学习更复杂的优化算法奠定了基础。

----
==微语录：穿过虚假的战场、现实的巨浪，活出自己喜欢的模样。——《浪浪山的小妖怪》==
