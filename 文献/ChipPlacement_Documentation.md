# Chip Placement 问题文档

## 1. 现实场景

### 1.1 问题背景

**Chip Placement（芯片布局）** 是超大规模集成电路（VLSI）物理设计流程中的核心环节。在芯片设计流程中，前端逻辑综合后得到的网表（netlist）需要映射到物理芯片区域上的具体位置，这一过程直接影响芯片的：

- **性能**：信号传输延迟
- **面积**：布线资源占用
- **功耗**：连线长度决定动态功耗
- **可制造性**：拥塞程度影响布线成功率

### 1.2 工业痛点

现代 ASIC 芯片设计面临以下挑战：

| 挑战 | 描述 |
|------|------|
| **规模庞大** | 单颗芯片包含百万至千万级 cell |
| **多目标冲突** | 线长、延迟、拥塞三者相互制约 |
| **复杂约束** | macro 位置、区域限制、密度约束 |
| **NP-hard** | 本质是二次指派问题（QAP）的扩展 |

### 1.3 传统方法局限

- **解析方法**（如力导向）：难以处理离散约束
- **启发式分割**：易陷入局部最优
- **商用 EDA 工具**：黑盒、不可定制

因此，将其建模为 **大规模稀疏约束多目标二进制优化问题（LSCMBO）**，利用进化算法求解，是当前研究热点。

---

## 2. 问题设定

### 2.1 决策变量

#### 2.1.1 定义

设芯片上有 $n$ 个电路模块（cells/macros），候选拼放网格有 $N$ 个 site，决策变量为二值矩阵：

$$X \in \{0,1\}^{n \times N}$$

其中：

$$x_{ij} = \begin{cases} 1, & \text{cell } i \text{ 被放置到 site } j \\ 0, & \text{否则} \end{cases}$$

#### 2.1.2 示例

|  | site1 | site2 | site3 | site4 |
|---|---|---|---|---|
| **cell1** | 0 | 1 | 0 | 0 |
| **cell2** | 0 | 0 | 0 | 1 |
| **cell3** | 1 | 0 | 0 | 0 |

表示：cell1 → site2，cell2 → site4，cell3 → site1。

#### 2.1.3 稀疏特性

由于约束 $\sum_j x_{ij} = 1$（每个 cell 只占一个 site），矩阵 $X$ 的非零元素数为 $n$，而变量总数 $D = nN$，稀疏率为：

$$\text{Sparsity} = 1 - \frac{n}{nN} = 1 - \frac{1}{N}$$

当 $N = 10^6$ 时，稀疏率高达 **99.9999%**，属于典型的**大规模稀疏优化问题**。

#### 2.1.4 变量规模

| 数据集 | $n$（cells） | $N$（sites） | $D = nN$（变量数） |
|--------|------|------|------|
| Small | 50 | 400 | 20,000 |
| Medium | 200 | 2,500 | 500,000 |
| 工业级 ISPD 2015 | ~$10^5$ | ~$10^6$ | ~$10^{11}$ |

---

### 2.2 目标函数

所有目标均采用**最小化**方向（PlatEMO 默认约定）。

#### 2.2.1 Objective 1: Wirelength（线长）

**物理含义**：所有信号连线的总长度，直接影响芯片面积、功耗和时序。

**计算方法**：采用 **HPWL（Half-Perimeter Wirelength）** 近似：

$$f_1(X) = \text{WL}(X) = \sum_{k \in \text{Nets}} \text{HPWL}_k$$

其中对每个 net $k$：

$$\text{HPWL}_k = (x_{\max} - x_{\min}) + (y_{\max} - y_{\min})$$

- $x_{\max}, x_{\min}$：net $k$ 所连接 cells 的 x 坐标极值
- $y_{\max}, y_{\min}$：y 坐标极值

**优化动机**：线长越短 → 芯片面积越小、功耗越低、信号完整性越好。

---

#### 2.2.2 Objective 2: Delay（延迟）

**物理含义**：芯片最长信号路径的传播延迟，决定芯片能跑的最高时钟频率。

**计算方法**：基于网表图 $G = (V, E)$ 的**最长路径**近似：

$$f_2(X) = \text{Delay}(X) = \max_{i, j} \text{dist}(i, j)$$

其中节点间边权为 cell 间的几何距离：

$$w_{ij} = \| \text{pos}_i - \text{pos}_j \|_2$$

使用 Floyd-Warshall 算法计算所有节点对的最长路径，取最大值作为关键路径延迟。

**优化动机**：延迟越小 → 芯片可运行频率越高、时序违例越少。

---

#### 2.2.3 Objective 3: Congestion（拥塞）

**物理含义**：布线资源需求超出供给的程度，拥塞过高会导致布线失败（DRC 错误）。

**计算方法**：对每个 site $r$，计算其包围的 net 数（routing demand $D_r$）与该 site 的布线容量（routing capacity $C_r$）之差：

$$f_3(X) = \text{Congestion}(X) = \sum_{r} \max(0, D_r - C_r)$$

其中：

- $D_r$：site $r$ 落在多少个 net 的 bounding box 内
- $C_r$：site $r$ 的路由容量（与位置相关，中心区域容量更高）

**优化动机**：拥塞越低 → 布线成功率越高、制造良率越高。

---

### 2.3 约束条件

#### 2.3.1 Constraint 1: 一一对应约束

每个 cell 必须恰好占据一个 site：

$$\forall i: \quad \sum_{j=1}^{N} x_{ij} = 1$$

违反值：$c_1 = \sum_i \left| \sum_j x_{ij} - 1 \right|$

#### 2.3.2 Constraint 2: 互斥约束

每个 site 最多只能被一个 cell 占据：

$$\forall j: \quad \sum_{i=1}^{n} x_{ij} \leq 1$$

违反值：$c_2 = \sum_j \max\left(0, \sum_i x_{ij} - 1\right)$

#### 2.3.3 Constraint 3: Macro 合法位置约束

大型 macro（如 SRAM、IP block）只能放置在不越界的合法位置。通过掩码矩阵 $M \in \{0,1\}^{n \times N}$ 实现：

$$M_{ij} = 0 \Rightarrow x_{ij} = 0$$

即若 site $j$ 无法容纳 macro $i$（如会超出芯片边界），则该决策变量强制为 0。

#### 2.3.4 总约束违反值

$$\text{CV}(X) = c_1 + c_2$$

在 PlatEMO 中作为 `PopCon` 输出，用于可行解优先的选择策略。

---

### 2.4 问题类型标签

```
<multi> <binary> <large/none> <constrained/none> <sparse/none>
```

- **multi**：多目标优化（3 个目标）
- **binary**：二值决策变量
- **large**：大规模变量（$D \geq 10^4$）
- **constrained**：含约束
- **sparse**：稀疏解结构

---

## 3. 数据集说明

### 3.1 数据来源

本项目使用合成数据集（结构与真实 ISPD 2015 / OpenROAD 数据一致）：

| 文件 | 作用 |
|------|------|
| [Prepare_ChipPlacement_Dataset.m](file:///c:/Users/User/Desktop/PlatEMO-GSNN-continuous/PlatEMO/Problems/Multi-objective%20optimization/Real-world%20MOPs/Prepare_ChipPlacement_Dataset.m) | 生成合成数据 |
| `Dataset_ChipPlacement_Small.mat` | 50 cells × 400 sites × 100 nets |
| `Dataset_ChipPlacement_Medium.mat` | 200 cells × 2500 sites × 500 nets |

### 3.2 数据结构

| 字段 | 维度 | 含义 |
|------|------|------|
| `Cells` | $n \times 4$ | [id, width, height, type]（type: 1=标准单元, 2=macro） |
| `Sites` | $N \times 2$ | site 坐标 [x, y] |
| `Nets` | cell array | 每个 net 连接的 cell 索引列表 |
| `MacroMask` | $n \times N$ | macro 合法位置掩码 |
| `SiteCapacity` | $N \times 1$ | 每个 site 的路由容量 |

---

## 4. 问题文件

### 4.1 实现文件

| 文件 | 数据集 |
|------|--------|
| [Sparse_ChipPlacement_Small.m](file:///c:/Users/User/Desktop/PlatEMO-GSNN-continuous/PlatEMO/Problems/Multi-objective%20optimization/Real-world%20MOPs/Sparse_ChipPlacement_Small.m) | Small（50 cells） |
| [Sparse_ChipPlacement_Medium.m](file:///c:/Users/User/Desktop/PlatEMO-GSNN-continuous/PlatEMO/Problems/Multi-objective%20optimization/Real-world%20MOPs/Sparse_ChipPlacement_Medium.m) | Medium（200 cells） |

### 4.2 使用方式

```matlab
% Small 数据集
platemo('problem', @Sparse_ChipPlacement_Small, 'algorithm', @NSGAII, 'maxFE', 1000)

% Medium 数据集
platemo('problem', @Sparse_ChipPlacement_Medium, 'algorithm', @NSGAII, 'maxFE', 1000)
```

---

## 5. 参考文献

1. **ISPD 2015 Placement Benchmark**: 经典 EDA 公开 benchmark，用于 global placement、wirelength optimization、congestion optimization 研究。
2. **OpenROAD Project**: 开源 RTL-to-GDS 芯片设计流程，提供现代工业级数据。  
   - 官网: https://theopenroadproject.org/
   - 代码: https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts
3. **PlatEMO 框架**: Ye Tian, Ran Cheng, Xingyi Zhang, Yaochu Jin. *PlatEMO: A MATLAB Platform for Evolutionary Multi-Objective Optimization*. IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87.
