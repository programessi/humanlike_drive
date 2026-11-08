# 层1逐例证据

案例与源目录由脚本显式指定，不根据结果自动挑选。局部／负载沿用v2，完整±180°、负90°、目标标定扰动及诊断采用v3。两版差别是接管前最多5秒有界等待；源码与实际配置随每条轨迹保存。

负载边界和扰动属于工程探测，失败不删除。无接触负对照应当跟踪失败；不能把它算作应通过的任务。

| 案例 | 版本 | 物理门槛 | 失败项 | 原始证据 |
| --- | --- | --- | --- | --- |
| dual_hold | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks/dual_hold/evaluation.json) |
| positive15 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks/positive15/evaluation.json) |
| negative15 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks/negative15/evaluation.json) |
| positive30 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks/positive30/evaluation.json) |
| negative30 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/negative30/evaluation.json) |
| positive90 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/positive90/evaluation.json) |
| negative90 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/negative90/evaluation.json) |
| positive180 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/core/positive180/evaluation.json) |
| negative180 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/core/negative180/evaluation.json) |
| crosszero_+1 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/crosszero_+1/evaluation.json) |
| crosszero_-1 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/crosszero_-1/evaluation.json) |
| hold_left_-0.05 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_left_-0.05/evaluation.json) |
| hold_left_+0.05 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_left_+0.05/evaluation.json) |
| hold_left_-0.10 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_left_-0.10/evaluation.json) |
| hold_left_+0.10 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_left_+0.10/evaluation.json) |
| hold_left_-0.20 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_left_-0.20/evaluation.json) |
| hold_left_+0.20 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_left_+0.20/evaluation.json) |
| hold_left_-0.40 | palm_primary_l1_v2 | 失败 | final_error_le_2deg, all_stop_tails_le_2deg, palm_contact_95pct, palm_contact_gap, palm_force_bounded, whole_hand_force_bounded, palm_slip_speed, palm_slip_distance | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_left_-0.40/evaluation.json) |
| hold_left_+0.40 | palm_primary_l1_v2 | 失败 | palm_slip_speed | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_left_+0.40/evaluation.json) |
| hold_right_-0.05 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_right_-0.05/evaluation.json) |
| hold_right_+0.05 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_right_+0.05/evaluation.json) |
| hold_right_-0.10 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_right_-0.10/evaluation.json) |
| hold_right_+0.10 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_right_+0.10/evaluation.json) |
| hold_right_-0.20 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_right_-0.20/evaluation.json) |
| hold_right_+0.20 | palm_primary_l1_v2 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_right_+0.20/evaluation.json) |
| hold_right_-0.40 | palm_primary_l1_v2 | 失败 | palm_slip_speed | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_right_-0.40/evaluation.json) |
| hold_right_+0.40 | palm_primary_l1_v2 | 失败 | final_error_le_2deg, all_stop_tails_le_2deg, palm_contact_95pct, palm_contact_gap, palm_force_bounded, whole_hand_force_bounded, palm_slip_speed, palm_slip_distance | [evaluation](../../../outputs/physical_feasibility/l1_v2_20261107/checks_parallel/hold_right_+0.40/evaluation.json) |
| perturb_friction_seed0 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_friction_seed0/evaluation.json) |
| perturb_friction_seed1 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_friction_seed1/evaluation.json) |
| perturb_friction_seed2 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_friction_seed2/evaluation.json) |
| perturb_friction_seed3 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_friction_seed3/evaluation.json) |
| perturb_friction_seed4 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_friction_seed4/evaluation.json) |
| perturb_load_seed0 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_load_seed0/evaluation.json) |
| perturb_load_seed1 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_load_seed1/evaluation.json) |
| perturb_load_seed2 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_load_seed2/evaluation.json) |
| perturb_load_seed3 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_load_seed3/evaluation.json) |
| perturb_load_seed4 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_load_seed4/evaluation.json) |
| perturb_combined_seed0 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_combined_seed0/evaluation.json) |
| perturb_combined_seed1 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_combined_seed1/evaluation.json) |
| perturb_combined_seed2 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_combined_seed2/evaluation.json) |
| perturb_combined_seed3 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_combined_seed3/evaluation.json) |
| perturb_combined_seed4 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/perturb_combined_seed4/evaluation.json) |
| fingers_less_+1 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/fingers_less_+1/evaluation.json) |
| fingers_less_-1 | palm_primary_l1_v3 | 通过 | — | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/fingers_less_-1/evaluation.json) |
| no_wheel_contact | palm_primary_l1_v3 | 失败 | joint_limits, motion_RMSE_le_2deg, final_error_le_2deg, all_stop_tails_le_2deg, palm_contact_95pct, palm_contact_gap, palm_friction_primary, palm_friction_primary_by_activity | [evaluation](../../../outputs/physical_feasibility/l1_v3_20261107/remaining/no_wheel_contact/evaluation.json) |
