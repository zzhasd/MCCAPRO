# MCCA-PRO：ICRA A 级审稿与算法创新路线

> 审计日期：2026-08-21。本文档区分“当前代码已经做到的事实”和“建议新增、尚待实验验证的研究命题”。任何尚未实现或验证的内容均不得写成论文既成贡献。

> **代码结构更新**：可提交实现已收敛为自包含 `mainline.py`、公共实验设施 `benchmark_MCCA.py`，以及 `labA_fact_mcca_scaling.py`、`labB_lsmcpp_comparison.py`、`labC_fact_certificates.py`、`labD_path_planning.py` 四个一键实验。本文后文出现的 `hybrid_mcpp.py`、`fact_mcpp.py` 和 `contracted_fact_lns.py` 是重构前的历史模块名，其实现现已合并进 `mainline.py`。

## 1. 审稿结论

升级后的代码已不再只是三阶段工程串联：它包含 FACT-MCPP 联合问题、原问题一树下界/双树可执行上界、可变动作库、root/depot/portal 可行性、真实 makespan、转向时间、联合三束搜索和动态 depot-cut 分离。尽管如此，**按目前证据仍不能达到 ICRA A（Definitely Accept）标准**，内部投稿准备度约为 B-。ICRA 2027 官方定义的 A 是“accepted ICRA papers 中前 15% 的 excellent paper，审稿人愿意为接收据理力争”。相对 LS-MCPP 的 fixed-footprint 30 种子优势已经统计显著，5×5 exact oracle 的证书也被动态割明显压缩；当前主要否决项变为：可变 footprint 主问题没有直接外部强基线、只有一个目标口径有效的官方地图、尚无标准地图族/大机器人规模/冲突消解/真实 footprint 平台验证，而且动态割与联合 LNS 的增益尚未在更大证书集上验证。

ICRA 2027 的投稿截止日期为 2026-09-15，论文总长度为 8 页（含参考文献），采用双匿名审稿；可提交视频，但审稿人没有义务查看论文外材料。官方信息：

- [ICRA 2027 Call for Papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)
- [IEEE RAS ICRA Reviewer Guidelines](https://www.ieee-ras.org/conferences-workshops/fully-sponsored/icra/information-for-icra-reviewers/)

## 2. 原始 E²TR-MCPP 的事实基线（升级前审计）

本节至第 5 节记录收到项目时的拒稿风险，用于解释后续修改来源；第 6 节以后才是当前升级版本。不要把本节中的“当前代码”理解为最终工作树状态。

原始方法 E²TR-MCPP 包含：

1. 加性权重测地 Voronoi 分区与边界格转移；
2. 多启动矩形精确覆盖启发式与局部 MILP；
3. 砖中心安全路网、度量闭包、最近邻和 2-opt 巡回。

其优点是实现完整、依赖少、在固定 5 机器人和 10% 独立点障碍上速度快，且已检查分区连通与铺砖精确覆盖。额外压力测试在 `N∈{32,40,60}`、机器人数量 `K∈{2,3,5,8,12}`、18% 障碍、12 个种子共 180 个实例上没有发现分区断连。

但压力测试也表明，`balance_tolerance=0.01` 不是算法保证：在扩展测试中区域尺寸极差最高达到 89 个格；在 `N=32,K=20` 时面积 CV 最高 0.1171，在 `N=100,K=20` 时最高 0.0769。代码只能在找到合适加权图或局部安全转移可行时接近 ε 均衡，不能把“ε-balanced”写成无条件性质。

### 2.1 科学有效性修正后的初步重跑

修正 rooted 责任域、统一 portal 安全路网、加入 depot 往返、每砖单位服务时间和 makespan 后，原 5 个尺度 × 3 个种子的 15 组配对实验全部通过 exact-cover、连通、root ownership、路线安全与 depot 闭环验证。混合方法相对自建基线的 makespan 在 15/15 例下降，配对平均下降 41.46%，中位下降 41.89%。

这个数字**不能作为论文主结果**：15 个基线实例中有 7 个 Lloyd 分区丢失原始机器人根，代码为了得到可执行比较而回退到 root-fixed geodesic Voronoi；其中 200×200 的 3 个基线全部触发回退，导致尺度越大表观 makespan 改善越高。它证明了修正后管线可运行，却同时证明旧基线过弱。原始数据位于 `results_rooted_makespan/benchmark_raw.csv`。

## 3. 会导致拒稿的关键问题

### 3.1 原始版优化目标错位

标准离线 MCPP 的主目标通常是任务完成时间（makespan）

\[
\min \max_r C_r,
\]

而原始代码主要比较总砖数、总路径长度和三个 CV。CV 很低不等于最大完成时间小，总长度小也不等于最后一台机器人更早完成。MIP-MCPP、LS-MCPP、MCFS 等强基线都直接报告 makespan；该问题现已修正。

### 3.2 三阶段在大图上仍是弱耦合

所谓“分区—铺砖耦合”只在 `N≤30` 的候选池里调用精确铺砖；大图只用面积 CV 和边界长度选择分区。路径代价没有参与分区或铺砖，最少砖数也未必产生最短巡回。因此目前不能声称联合优化。

### 3.3 执行代价模型不完整

- 机器人初始位置到首个砖中心、以及返回初始位置的代价没有计入输出路线；初始位置只用于选择巡回起点。
- 没有定义一次“铺砖/覆盖动作”的服务时间，因而砖数与路径长度无法合成为完成时间。
- 没有考虑转向、加减速、非完整约束或多机器人时空冲突。
- `validate()` 不验证路线段是否在自由空间内，也不检查路线是否从机器人真实起点出发并闭合返回。

### 3.4 基线比较可能不公平

基线 MST 的边权是相邻矩形中心的直接欧氏距离，输出也直接连接两个中心。两个只共享部分边界的矩形，其中心连线不一定落在二者并集内，因此基线折线可能穿过未分配区域或障碍。升级方法则通过共享边界 portal 绕行。两者必须使用相同的安全路网和相同的 depot 代价后再比较。

### 3.5 复杂度与实验外推不足

- `route_tsp()` 为每个机器人建立稠密全对最短路矩阵，时间/内存随砖数近似二次增长。
- `N≤30` 的分区候选评估反复调用 MILP，却不受 `tiling_time_limit` 的全局预算约束。
- 当前只有随机独立点障碍、3 个种子、固定 5 台机器人，无标准地图、无强公开基线、无硬件或高保真仿真。

## 4. 相关工作差异矩阵

| 工作 | 核心抽象 | 主要目标 | 与本项目的关系/威胁 |
|---|---|---|---|
| [DARP](https://doi.org/10.1007/s10846-016-0461-x) | 等面积、含根、连通分区 | 面积平衡 | 当前加权分区若没有下游真实代价，只是同类增强 |
| [SCoPP](https://arxiv.org/abs/2103.14709) | 可扩展聚类、冲突格拍卖、NN 路径 | 任务时间/负载 | 已覆盖“可扩展分区 + 最近邻路径”叙事 |
| [MSTC*](https://arxiv.org/abs/2108.04632) | 物理约束下的多机器人 STC | makespan | 要求本项目说明可变矩形动作带来的本质新问题 |
| [MIP-MCPP](https://arxiv.org/abs/2306.17609) | Min-Max Rooted Tree Cover 的 MIP | makespan，4-近似覆盖路线 | 已占据“根式树覆盖 + MIP + 理论界”位置 |
| [LS-MCPP / ESTC](https://arxiv.org/abs/2312.10797) | 直接在分解图上做路径局部搜索 | makespan | 仅增加边界局部搜索或 MAPF 后处理不够新 |
| [TMSTC*](https://arxiv.org/abs/2212.02231) | minimum-brick 分解、brick adjacency 与 turn-minimizing STC | 转向次数 | “矩形/brick 分解减少转向”已有直接先例 |
| [Turn-minimizing MCPP, ICRA 2019](https://eprints.whiterose.ac.uk/id/eprint/143336/) | rank 分解 + 多旅行商分配 | 转向与覆盖时间 | 单独优化转向拓扑不是新的主问题 |
| [Near-optimal turn-cost coverage](https://arxiv.org/abs/2310.20340) | atomic strips、LP cycle cover 与 tour connection | 带转向覆盖成本与下界 | 朝向提升或 turn-aware TSP 必须提供更具体的新结构 |
| [Path deconfliction](https://arxiv.org/abs/2411.01707) | 任意部分障碍网格、MAPF 去冲突 | makespan 与可执行性 | 已系统处理网格缺损、转向和多机器人冲突 |
| [MCFS](https://arxiv.org/abs/2403.13311) | 等值线图 + MMRTC + 平滑连续覆盖 | makespan、曲率、重叠 | 说明只优化网格距离和中心 TSP 的机器人真实性偏弱 |
| [Multi-CAP](https://arxiv.org/abs/2509.14941) | 未知环境的连通子区 + VRP | 在线覆盖时间 | “子区图 + VRP”本身已不是新贡献 |
| [Generalized Covering Salesman](https://doi.org/10.1287/ijoc.1110.0480) | 选择覆盖节点并巡回 | 访问与覆盖联合成本 | 说明“先选覆盖点再 TSP”在运筹学上已有成熟抽象 |
| [Minimum Connected Set Cover](https://arxiv.org/abs/2504.07725) | 选择诱导连通的覆盖集合 | 覆盖成本 | 统一模型必须明确区别：精确无重叠、多个根、min-max、几何 portal |

## 5. 已淘汰的创新候选

| 候选 | 可落地性 | 创新深度 | 决策 |
|---|---:|---:|---|
| 更复杂的 power diagram 权重更新 | 高 | 低 | 淘汰：仍按面积而非执行代价分区 |
| 增加 MILP 窗口、随机重启或换元启发式 | 高 | 低 | 淘汰：典型工程增强 |
| NN/2-opt 换成 LKH/遗传算法 | 高 | 低 | 淘汰：求解器替换，不能形成机器人学贡献 |
| 学习边界代价或强化学习分区 | 中 | 低至中 | 淘汰：需要数据，缺少可解释保证，且没有解决问题建模缺口 |
| 单独增加 MAPF/动态避障 | 中 | 中 | 不作为核心：是必要执行层，但 LS-MCPP 已有直接先例 |
| 固定访问序上的朝向提升最短路 | 高 | 中 | 不作为核心：模型正确但 `floor_medium` 长预算下只额外改善 0.46%，且转向覆盖已有充分先例 |
| 只用 tile count 代替 area 做均衡 | 高 | 低 | 淘汰：仍忽略路由和 depot，且服务模型不完整 |
| 全局联合 MILP | 低至中 | 高 | 保留为小图精确解/下界，不单独作为可扩展主算法 |
| 统一树覆盖模型 + 联合大邻域精确重优化 | 中 | 高 | **保留为主方向** |

## 6. 建议核心问题：FACT-MCPP

建议把论文核心定义为 **Footprint-Aware Coupled Tree-Cover Multi-Robot Coverage Path Planning（FACT-MCPP）**。

给定自由栅格图 \(G=(V,E)\)、机器人根位置 \(q_r\)、合法各向异性覆盖动作集合 \(\mathcal P\)。动作 \(p\) 覆盖矩形格集合 \(C_p\subseteq V\)，中心为 \(c_p\)，服务时间为 \(s_p\)。选择每台机器人的动作集合 \(P_r\) 和闭合安全路线 \(\pi_r\)，满足：

1. 精确覆盖：每个自由格恰由一个被选动作覆盖；
2. 根连通：同一机器人选中的矩形在共享边界 portal 图上与包含 \(q_r\) 的根矩形连通；
3. 路线从 \(q_r\) 出发、访问所有选中动作中心并返回 \(q_r\)；
4. 最小化真实任务完成时间：

\[
\min \; Z=\max_r\left(\sum_{p\in P_r}s_p+\frac{\ell(\pi_r)}{v_r}\right).
\]

当只允许 \(1\times1\) 动作时，该问题退化为标准根式网格 MCPP，因此至少与标准 MCPP 一样困难。这一问题定义把本项目独有的“多尺度各向异性矩形动作”变成第一等公民，而不是铺砖阶段的工程细节。

## 7. 可证明的根式树上下界

先把可执行路线的离散语义钉死：节点是 depot 与被选动作中心；相邻矩形通过共享边界 portal 相连，边几何是“中心—portal—中心”；路线必须是该安全 portal 图上的闭合 walk，或者是其最短路度量闭包中的等价巡回。只有在这一共同路网中，下面的树—巡回关系才成立；如果允许连续空间中任意穿越矩形而树仍强制经过每个中间矩形中心，原来的 2-近似证明并不成立。

对每台机器人，在 depot、被选矩形中心和安全 portal 边上选择根式生成树 \(T_r\)。对同一组 exact-cover、assignment 和 tree 约束，分别定义

\[
Z_L=\max_r\left(\sum_{p\in P_r}s_p+w(T_r)/v_r\right),
\qquad
Z_U=\max_r\left(\sum_{p\in P_r}s_p+2w(T_r)/v_r\right).
\]

这里 \(Z_L\) 是原始闭合 portal-walk 问题的松弛，\(Z_U\) 是可执行双树代理。代码中的 `solve_exact_fact_bounds()` 独立求解两个联合模型，而不是把一个 surrogate 的 MIP gap 误写成原问题 gap。

### 命题 1：下界

任一可行闭合 walk 的边支撑连通并包含所有被选动作中心及 depot，从中删除环可得一棵不更重的根式生成树。因此，全局最优一树模型 \(Z_L^*\) 满足

\[
Z_L^*\le Z^*.
\]

若一树 MILP 未在时限内证明最优，求解器的全局 dual bound 仍是 \(Z_L^*\) 的下界，因而也是原始 FACT-MCPP 的合法下界。

### 命题 2：可行上界

将每棵 \(T_r\) 的边遍历两次，再在安全度量闭包中 shortcut，可得到从 depot 出发并返回、访问全部动作中心的无碰路线。2-opt 只接受降成本交换，因此不会破坏该上界。

### 命题 3：全局精确双树模型的 2-近似界

对一树模型的任一最优解，非负服务时间给出

\[
Z_U^*\le 2Z_L^*\le 2Z^*.
\]

树加倍执行路线的 makespan 不超过 \(Z_U^*\)，所以若双树联合模型被精确求解，就得到 FACT-MCPP 的 2-近似解。即使模型截断，也可以直接报告“当前可执行 incumbent / 一树模型 dual bound”作为端到端证书，而不是只报告 MILP 内部 gap。

该界只覆盖“服务时间 + portal 平移时间”的度量模型。代码现已支持 `turn_time_90`，并将任意 portal 折线的最小包角旋转时间纳入真实 makespan 和候选验收；但带转向目标不满足上述纯度量 shortcut 证明，不能沿用 2-近似宣称。

当前回归证据：4×4 无障碍、2 台机器人与一个 5×5 中心单障碍、2 台机器人实例均把上下模型求至 0 gap；后者原问题下界为 7.0322，可执行上界为 11.0645，证书比为 1.5734。额外 6 例随机小图由 `benchmark_fact_certificates.py` 复现：4×4 的 3/3 例在 12 s/模型内最优，证书比均约 1.586。未加割时，5×5 的 0/3 例在该时限内证明最优，合法证书比为 1.763–2.314，说明单商品流松弛很弱。

代码现已实现 root-LP 上的动态 depot-cut 分离。对机器人 \(r\)、不含人工 depot source 的 placement 子集 \(S\) 和 \(p\in S\)，加入

\[
\sum_{e\in\delta(S)}y_{er}+\sum_{q\in S\cap R_r}z_{qr}\ge z_{pr},
\]

其中 \(R_r\) 是能覆盖 depot 的 root-action 候选。若 \(p\) 被选且已选 root action 不在 \(S\)，任何可行连通树必须有跨割边；若 root action 在 \(S\)，第二项使不等式成立，故该割不删除任何整数可行解。以当前 LP 的 \(y,z\) 为容量，在人工 source 到每个分数 \(p\) 之间求最小割即可分离最违反约束。5×5 三种子、12 s/模型下，仅加入约 37–56 条动态割，平均证书比从 2.044 降到 1.805；三个种子分别为 2.055→1.958、2.314→1.735、1.763→1.722。最弱 seed 1 的一树 gap 从 46.7% 降到 6.0%，双树 gap 从 39.1% 降到 8.4%。这明显优于静态枚举 3902 条 singleton/pair cuts；但 6×6 单例在相同总预算下从 1.738 恶化到 1.784，说明分离时间开始挤压 branch-and-bound，仍不能称为可扩展求解已经解决。

这一理论角度比当前“NN + 2-opt 通常更短”更强：它提供可行性和质量界，同时允许 NN/LKH 作为只改善不破坏保证的后处理。

## 8. 联合 MILP 与可扩展求解

### 8.1 小图精确模型

变量：

- \(z_{pr}\in\{0,1\}\)：动作 \(p\) 是否分配给机器人 \(r\)；
- \(y_{er}\in\{0,1\}\)：兼容矩形邻接边 \(e\) 是否进入机器人 \(r\) 的根式树；
- \(f_{er}\ge0\)：单商品根流，用于保证所有已选动作连接到 depot；
- \(Z\ge0\)：makespan。

核心约束为逐格 exact-cover、边—节点一致性、根流守恒和每机器人代价不超过 \(Z\)。正边权使最优连通子图自然无环；也可显式加入边数约束。该模型用于：

1. 小实例最优解；
2. LP/MIP 下界与最优性 gap；
3. 验证可扩展算法是否真的接近联合最优，而不是只胜过弱基线。

### 8.2 主算法：Bottleneck-guided Joint Large Neighborhood Search

1. 用现有加权测地分区和混合铺砖产生可行 warm start；
2. 以真实 makespan 最大的机器人为 bottleneck，而不是以面积偏差最大者为目标；
3. 选择 bottleneck 与相邻机器人边界的宽度 \(b\) 带状区域；
4. 冻结两侧内部动作和树枝，只对边界带联合求解“格归属 + 矩形动作 + portal 树边”；
5. 用全局真实 makespan 重新评价，仅接受严格改善；
6. 对所得树做 tree-doubling shortcut 和 2-opt；重复到无改善或预算耗尽。

该邻域一次改变三个当前互相割裂的决定，因此不是简单边界格交换。每次严格降低有限解空间上的真实目标，所以在有限步内终止于该联合邻域的局部最优。对每次局部 MILP 保存 dual bound，可报告局部最优性证书。

### 8.3 当前落地版本：三束保底搜索

代码中的 `refine_coupled_beam()` 同时维护三条从同一 incumbent 和同一随机状态出发的搜索束：

1. `assignment_geometry`：按几何边界排序，转移完整覆盖动作并重算两台机器人的安全路线；
2. `route_aware_geometry`：额外生成多个局部 exact-cover 铺砖方案，不按砖数直接接受，而按完整 depot-aware makespan 选择；
3. `dual_guided_fact`：用 exact-cover LP 对偶价格和路线边际代价选择边界动作，并允许 frozen-anchor FACT-MCPP 联合决定补丁格归属、矩形动作和 rooted portal tree。

三束都只接受全局 makespan 严格下降且通过 rooted connectivity、exact cover、portal 路线安全和 depot 闭环验证的解，最终返回 makespan 最低的 incumbent。因此对偶或 FACT 邻域只能带来可选增益，不会因改变搜索轨迹而让最终结果弱于内部保守束。

对边界动作 (p:r\rightarrow s)，可扩展束使用的联合价格由四部分组成：

\[
g(p,r,s)=
\sum_{c\in C_p}\pi_{r,c}
-|C_p|\bar\pi_s
+\Delta^-_{r}(p)-\underline{\Delta}^{+}_{s}(p),
\]

其中 \(\pi_{r,c}\) 是机器人 \(r\) 的 exact-cover LP 等式约束对偶价格；\(\Delta^-_r(p)\) 是从当前安全度量巡回删除动作中心的精确 detour；\(\underline{\Delta}^{+}_s(p)\) 是将该中心插入接收方巡回的欧氏下界。LP 对偶只在固定候选矩阵的局部敏感性意义下成立；区域变化会改变合法矩形集合，因此 \(g\) 是候选排序价格，不是全局 Benders cut。所有动作仍由真实 makespan 重新验收。

最终组件消融（20×20、3 台机器人、15% 障碍、10 个种子、每束 5 s）为：

| 变体 | makespan 相对初始解平均改善 | 获胜实例 | 平均总运行时间 |
|---|---:|---:|---:|
| assignment-only | 7.46% | 9/10 | 0.98 s |
| route-aware，无对偶/FACT | 9.08% | 9/10 | 2.22 s |
| dual + route-aware + FACT 三束 | **10.60%** | **10/10** | 7.51 s |

在这 10 例中，最终选中 assignment、route-aware、dual-guided 束的次数分别为 2、4、4；只有 1 次最终接受移动直接来自局部 FACT-tree MILP，其余 dual 束收益来自联合价格排序。一个两个保守束均停滞的实例由 dual 束获得 8.78% 改善。

尺度消融（20/50/100、5 台机器人、每尺度 3 个种子、每束 5 s）中，assignment、route-aware、完整三束平均改善分别为 6.82%、6.86%、**7.49%**，9/9 实例全部可行且完整方法全部改善。最终只有 1/9 选择 dual 束，且 100×100 均由保守束胜出；在每束 30 s、固定最多 4 次迭代的定向实验中，100×100 的 3 个种子分别由 dual、route-aware、assignment 束胜出。由此只能声称联合价格提供互补候选，不能声称它在所有尺度占主导。

### 8.4 官方强基线：fixed-footprint 特例已显著领先，但外推仍受限

已增加 `tile_shapes` 实例参数；设置 `tile_shapes=()` 时只允许 1×1 动作，服务时间设为 0，FACT 严格退化为 fixed-footprint、纯路径 MCPP。`benchmark_lsmcpp_adapter.py` 在独立环境中运行[官方 LS-MCPP 仓库](https://github.com/reso1/LS-MCPP)的 MFC 初始化 + 3000 次 local-search 配置，不复制或修改其 GPLv3 源码。两边使用相同细栅格、障碍、机器人根和闭环路径；适配器显式验证 LS-MCPP 的完整覆盖、相邻边合法与 depot 闭环。

20×20、3 台机器人、10% 随机点障碍、seeds 0–29、MCCA 最多 10 次边界迭代、每次 12 个候选、每束 20 s 上限的结果由 `labB_lsmcpp_comparison.py --profile paper` 重新生成到带时间戳的实验目录。固定 1×1 足迹时没有可重铺的替代动作或 FACT patch，因此代码只运行不重复的 `assignment_geometry` 束；昂贵的朝向状态束默认关闭并单独消融：

| 同口径目标 | MCCA 相对 LS-MCPP 平均改善 | 胜/平/负 | 单侧 Wilcoxon | paired mean difference bootstrap 95% CI |
|---|---:|---:|---:|---:|
| 纯平移长度 | **4.03%** | 23/3/4 | 1.91×10⁻⁴ | [3.27, 8.60] |
| 初始朝北、90° 转向时间 0.5 | **3.30%** | 22/1/7 | 3.84×10⁻⁴ | [3.25, 8.77] |

纯长度与转向配置平均运行时间分别为 1.69 s 和 3.25 s，LS-MCPP 为 4.65 s；30/30 双方路径均通过上述结构检查。统计证据支持该 fixed-footprint 随机分布上稳定优于 LS-MCPP，但含转向仍有 7 个负例，且实验尚未调用 LS-MCPP 的 PBS path-deconfliction 后处理，不能外推到多机器人无冲突执行时间。

在 LS-MCPP 自带的 `floor_medium` 官方实例（40×40、1296 自由格、8 台机器人）上，统一搜索预算后，MCCA 纯长度 makespan 为 192，LS-MCPP 为 212，改善 9.43%；相同“初始朝北、90°=0.5”目标下，MCCA 为 219.5，LS-MCPP 为 237.5，改善 7.58%。MCCA 两项运行时间为 6.50/7.74 s，LS-MCPP 为 19.82 s；该结果由 `labB_lsmcpp_comparison.py --profile paper` 重新生成。早先 2 s/束得到的 246.5 是预算不足造成的假阴性，已废弃。

官方实例适配器现在硬性拒绝不同目标：`terrain_*` 使用随机加权边，而当前 MCCA 使用单位栅格代价；`terrain_large` 与 `floor_large` 还含重复 depot。前者若直接比较会混合目标函数，后者不满足本问题“每机器人唯一根”的输入假设，因此均未纳入结果。当前只有一个有效官方地图，仍远不足以声称跨分布泛化。

朝向提升消融采用固定 singleton 访问序上的 “(cell, arrival heading)” 乘积图最短路，并用 4 状态 DP 耦合相邻 leg；对给定顺序它是精确的，且普通 metric-expansion 路线始终保留为 incumbent。`floor_medium` 在 120 s/10 次长预算下，普通联合搜索从 255.5 降到 219.5，加入朝向精确实现后从 253.5 降到 218.5，只额外贡献 1.0（0.46%）且计算显著更贵。因此该方向已被否决为核心创新，只保留为可选 route realization oracle。

### 8.5 相邻工作碰撞审计：创新主张必须降噪

进一步检索后，以下元素都不能单独声称新颖：

- [LS-MCPP（AAAI 2024）](https://ojs.aaai.org/index.php/AAAI/article/download/29707/31213) 已在分解图上用 grow、deduplicate、exchange 三类边界编辑算子，按真实 coverage-path makespan 验收；“边界交换 + 路由重算”不是本工作的独立创新。
- [MIP-MCPP（RA-L 2023 / SoCS 摘要）](https://ojs.aaai.org/index.php/SOCS/article/download/31585/33745/35642) 已联合优化多根 tree cover 并讨论 STC 近似界；“rooted tree MILP”本身不是新贡献。
- [Covering Salesman branch-and-cut](https://arxiv.org/abs/2104.01173) 与 [Capacitated Covering Salesman](https://arxiv.org/abs/2403.06995) 已把“选择能覆盖需求的服务节点/邻域”和 routing 联合起来；“集合覆盖 + TSP”不是新贡献。
- 在车辆路径中，集合划分 LP 对偶用于 pricing/reduced-cost 筛选是经典方法；例如 [set-partitioning VRP 研究](https://commons.case.edu/wsom-ops-reports/517/) 已明确使用最优对偶变量生成列。因此当前 exact-cover dual 只能称为 domain-specific candidate price，不能称为新分解理论或 Benders cut。
- [Min-max VRP column generation](https://doi.org/10.4230/OASIcs.ATMOS.2013.137) 已表明路线列生成可为 min-max 目标产生强下界。因此下一阶段即使实现 rooted-action-tree Dantzig–Wolfe，也不能把“min-max + column generation”本身写成创新；可辩护的新点必须是同时处理 exact non-overlap footprint actions、可变 root action 与 portal-tree pricing 的专用结构和可证明 reduced-cost oracle。
- [Multi-CAP](https://arxiv.org/abs/2509.14941) 已把连通子区、全局多仓 VRP 与局部覆盖结合，并有三机器人实机；“子区图 + VRP + 局部覆盖”的层级结构也不新。
- [TMSTC*](https://arxiv.org/abs/2212.02231) 已用 minimum bricks 与 bipartite-graph maximum independent set 组织转向友好的覆盖；[ICRA 2019 turn-minimizing MCPP](https://eprints.whiterose.ac.uk/id/eprint/143336/) 和 [ALENEX 2024 near-optimal turn-cost coverage](https://arxiv.org/abs/2310.20340) 进一步说明“显式减少转向”与 LP/strip/cycle-cover 结构已有先例。当前朝向 DP 只能称为 FACT 路线实现的受限精确子问题。

现阶段仍具有可辩护潜力、但尚未被“首次性”证明的，是以下**整体命题**：在同一个多机器人 min-max 模型中，联合选择互不重叠的可变矩形覆盖动作、根归属和 portal-tree，并由一树/双树两套模型给出原始安全闭合路线的上下界与可执行证书。论文主贡献应锁定这一结构；三束搜索和对偶候选价属于实现该模型的 scalable solver，而不是与主模型并列的理论创新。

### 8.6 必须避免的过度宣称

- 局部大邻域算法只有单调下降和有限终止，不自动继承全局 2-近似界；只有独立求解的一树/双树全局模型形成该证书。
- 除非系统检索进一步确认，不能写“首次提出”。可以写“we formulate”，并具体对比 fixed-footprint MMRTC、connected set cover 和 covering salesman。
- 只在随机点障碍上有效，不能声称对复杂环境普遍更优。
- 对偶价格不是全局分区目标的精确梯度，局部 FACT-tree 也不是大图性能的主要来源；论文必须分别消融并如实报告束选择频率。
- 不能把 LS-MCPP 官方含转向的 `tau` 与本方法纯长度直接比较；适配器现在分别报告 `lsmcpp_length_makespan` 和 `lsmcpp_turn_aware_makespan`。早期单例中由混用口径得到的约 22.9% 数字已作废。

## 9. 达到 A 级所需证据

### 必须完成

1. 实现 depot-aware makespan、服务时间、树加倍保证路线和路线几何验证；
2. 实现小图联合一树下界与双树上界，至少在几十到上百个小实例上报告原问题 certificate ratio，而非 surrogate gap；
3. 实现联合边界大邻域，并做分区/铺砖/树三个决策层的消融；
4. 使用相同安全路网公平重算现有基线；
5. 对比 DARP、MSTC*/MIP-MCPP、LS-MCPP/ESTC，若连续空间是主叙事则增加 MCFS；
6. 标准地图至少覆盖 open、maze、rooms、warehouse、窄通道和块状障碍；
7. 机器人数量至少 \(K\in\{2,5,10,20,50\}\)，每类足够随机种子并报告 95% CI、效应量、失败率；
8. 报告 makespan、总成本、服务/行驶分解、转向数、重复覆盖、规划时间、峰值内存和下界 gap；
9. 至少提供 ROS2/Gazebo 多机器人执行；竞争 A 级最好有 2–5 台真实机器人或与具体传感/作业装备一致的硬件闭环验证。

### A 级门槛（内部 go/no-go）

以下条件全部满足才把目标评为“A 级竞争力”，而不是保证录用：

- 核心方法相对最强公开基线在主要 makespan 指标上有稳定且统计显著的优势，而非只胜过原 BFS/MST；
- 联合模型消融证明收益确实来自跨阶段耦合，不是某个更强 TSP 求解器；
- 小图与最优解/下界的 gap 有竞争力，大图扩展到至少 256×256/100 robots 的一部分设置；
- 理论命题经过逐项可执行假设核验；
- 为矩形动作给出真实机器人学语义并标定 \(s_p\)：若不同面积动作仍统一使用常数服务时间，必须有“一次曝光/一次处理动作”的硬件依据，否则模型会被审稿人视为人为偏好大砖；
- 真实或高保真执行证明 footprint、depot、转向和冲突模型没有被离线指标掩盖；
- 所有结论都能在 8 页内形成单一清晰故事。

## 10. 当前最诚实的论文定位

当前升级代码已具备一篇“联合优化 + 可执行证书”论文的算法骨架，但现有证据仍不足以支持 ICRA A 级。论文应把贡献压缩为三点：

1. **新问题与统一抽象**：多机器人、多尺度各向异性动作、精确覆盖、根连通与 makespan 的 FACT-MCPP；
2. **理论与精确基准**：联合 rooted exact-cover 一树下界、双树可执行上界及精确求解时的 2-近似保证；
3. **可扩展算法与机器人验证**：bottleneck-guided 联合大邻域，在真实 makespan 上单调改善，并由小图下界、强基线和执行实验共同验证。

这条路线保持“分区—铺砖—避障巡回”的主线，但把创新焦点从三个常见组件各自升级，转为一个现有强工作没有直接覆盖的耦合决策结构。

## 11. 下一轮真正值得投入的算法问题

当前最需要解决的不是增加第四条启发式束，而是强化联合模型本身。5×5 随机图已经显示单商品流 formulation 的界很弱。优先级应为：

1. 为每个机器人加入由 depot cut / generalized subtour separation 得到的连接割，比较其对 root-flow LP gap 的压缩；
2. 研究按“一个机器人完整 rooted action-tree 为一列”的 min-max set-partitioning 主问题，pricing 同时考虑 cell dual、机器人 makespan dual 与 portal-tree reduced cost；
3. 只有当 pricing 能给出严格负 reduced-cost 证书或受控下界时，才把当前启发式联合价格升级为正式的 branch-price / logic-based decomposition；
4. 若上述方法不能在 8×8–12×12 明显扩大可证明规模，就保留现有双模型作为验证 oracle，不把 decomposition 作为论文主贡献。

这是本轮审计后的 go/no-go：统一 FACT 模型与上下界链具有论文级深度；对偶排序和三束搜索本身没有。要达到 ICRA A 级竞争力，下一阶段必须至少在“更强可证明求解”或“真实 footprint 机器人闭环”中完成一项硬突破，并同时补齐强基线与标准地图。

一次静态割强化已被实现并否决为默认方案：加入 singleton 与二动作 portal-cut 后，在 5×5、seed 0、10 s/模型上，一树 MIP gap 从 28.6% 降到 26.1%，但约束增加 3902 条，端到端证书比从 2.055 变差到 2.251。代码保留 `strengthen_connectivity`/`--connectivity-cuts` 作为负面对照，默认关闭。

动态 depot-cut 分离在 3 个 5×5 困难实例上平均把证书比改善 11.7%（2.044→1.805），但 6×6 seed 0 的 go/no-go 已出现反例：同为 12 s/模型，原始根流证书比 1.738，动态割因占用分支定界预算恶化到 1.784。因此它只保留为 5×5 exact oracle 的可选开关，默认不启用。下一轮不应继续堆叠 cut family，而应转向完整 rooted-action-tree 列生成，并报告 root-LP bound、pricing 最优性、列数与墙钟时间。即使动态割在小图稳定有效，它属于标准 cut separation 在新 FACT formulation 上的应用，不足以单独构成 A 级创新。

rooted-action-tree Dantzig–Wolfe 原型也已实现并完成第一轮证伪。每列是一台机器人的完整 rooted、内部无重叠 action tree；master 只保留逐格 exact-cover 和每机器人 makespan 行。由于每个机器人列必覆盖自己的 protected depot、其他机器人列禁止覆盖该格，depot exact-cover 行本身已蕴含 convexity，删除重复的 \(\sum_j\lambda_{rj}=1\) 行可减少对偶退化。pricing 是精确 prize-collecting rooted action-tree MIP；只有所有 pricing 被证明最优且无负 reduced-cost 时，接口才返回 `certified_lower_bound`，否则返回 NaN，绝不把受限主问题值误报为全局下界。

历史 3×3/4×4 column-generation go/no-go 显示：3×3 singleton、3×3 domino、4×4 singleton 均认证收敛；3×3 domino 的 Dantzig–Wolfe 下界仅比紧凑 root LP 强 0.73%，singleton 基本相同。4×4 domino 在 15 s 全局预算内不收敛；一次 224 s/60 轮审计中 master 已达到紧凑整数最优 8.7426，但仍存在 −0.0858 的严格负 reduced-cost 列，暴露出互补列缺失与严重退化。由于该方向已被否决且不属于最终主线，旧 CSV 不再随精简代码分发。结论是：该分解数学正确、可作为小图下界研究原型，但当前既没有显著界提升也没有可扩展性，不能作为 A 级核心。下一步转向 component-contracted FACT-LNS：把边界带外每个固定动作连通分量收缩为 mandatory supernode，在保留其内部树成本与全部 portal 接口的前提下，对边界带的归属、动作和连接树做精确联合重优化。
