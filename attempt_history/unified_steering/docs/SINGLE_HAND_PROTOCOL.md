# S00-B：单手支持、释放与受约束停放随机化

入口为`unified_steering.support`，配置为`configs/s00_support.yaml`。这是物理脚本诊断，尚无RL。

## 初态与自由手

左右手各有一个固定安全停放构型，通过静态IK生成。掌部参考位置相对轮轴坐标系为左`[-0.12,0.20,-0.04] m`、右`[-0.12,-0.20,-0.04] m`，朝向与名义抓握参考一致。实际接受条件检查整只手与轮盘的距离，以及自由手和肘部守卫到其他已启用碰撞几何的距离，不能只看腕部位置。

初态臂余量≥5°，停放手与轮盘间隙≥2 cm，自由手/肘守卫与其他相关几何间隙≥12 mm。释放路径按关节间隔≤0.005 rad预采样；手轮接触在释放初期允许，其他身体间隙与实际执行的碰撞/余量按500 Hz检查。该检查限于现有碰撞几何，不代表完整CAD覆盖。

诊断期间自由手由有限输出伺服保持停放构型。后续统一策略训练只使用这些区域生成初态，运行中的自由手仍由网络控制，不锁在区域内。

## 非零轮角的准备

在reset期间设置所需轮角，并通过一个明确记录的有限外部托持力矩帮助闭手。该准备夹具的PD为5 Nm/rad、0.3 Nm·s/rad，上限1 Nm，补偿当时回正项。它只作用于2秒物理准备过程。

测量阶段从准备结束的实际轮角起算，夹具全部撤除，外部扰动力矩为0；回正、阻尼和干摩擦保持。准备结束角度与请求角度的偏差、速度、接触、辅助力矩均记录，检查角度偏差≤2°、速度≤1°/s。禁止把辅助夹具引入策略环境任务步。

## 测试

- **保持**：左右支持×0°、±15°、±30°，固定关节目标3秒；自由手预先停放。
- **往返**：同10个条件，时间参考`0→+5→0→−5→0°`相对初始实际轮角，每段5秒、端点停0.5秒；峰值参考速度1.875°/s。脚本生成腕部圆弧和关节目标，仅用作物理诊断。
- **释放**：双手初态，3秒张开自由手、3秒移向停放构型、3秒释放后保持。相同10个条件。
- **随机初态**：名义保持通过后，左右各5个自由臂关节±2°随机扰动，经几何筛选后在+30°保持3秒。该扩展只验证自由臂停放初态，不宣称抓握/负载广泛泛化。

保持漂移门槛2°；往返RMSE门槛2°；释放全程误差峰值≤3°、释放后绝对误差≤2°。还检查真实离轮、支持接触、运行余量≥2°、无非正常碰撞、有限执行器输出。CSV按20 Hz保存，间隙/余量/误差峰值/力限等保留500 Hz区间统计。

## 开环与反馈对照

默认开环：关节参考只由时间与静态IK生成，不读取轮角误差。

`--feedback`明确启用另一种**脚本基线**：腕部参考相位加上`3×轮角误差`的修正，限幅±8°、修正速度≤4°/s。自由手释放路径不变，仍使用相同有限输出执行器，辅助夹具仍撤除。它用于区分开环精度不足与控制权限不足，不是学习策略，不会接入主线策略的执行层。

## 重现

从项目根目录运行，下列输出目录已存在，重跑应换新名称：

```bash
bash unified_steering/scripts/run.sh -m unified_steering.support \
  --stages hold motion release --output outputs/s00_support_full_v1
bash unified_steering/scripts/run.sh -m unified_steering.support \
  --feedback --stages motion release --output outputs/s00_support_feedback_v1
bash unified_steering/scripts/run.sh -m unified_steering.support \
  --stages jitter --nominal-report outputs/s00_support_full_v1/summary.json \
  --output outputs/s00_support_jitter_v1
bash unified_steering/scripts/run.sh -m unittest discover -s tests -v
```

每次运行拒绝覆盖输出，记录配置/资产哈希、初态和逐例门槛。jitter复用的名义保持证据必须匹配配置与场景。初态以`.initial.npz`保存；只有通过保持检查的状态进入精选初态库。

图表归档工具为`python -m unified_steering.support_report`，需要可选依赖`pip install -e './unified_steering[reports]'`。实验结论见[S00-B报告](../results/S00_SUPPORT.md)。
