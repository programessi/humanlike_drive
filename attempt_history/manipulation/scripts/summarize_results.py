"""Rebuild the v2 pilot comparison from saved results; never trains or simulates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs"
RUNS = ROOT / "runs"
CONTROLLERS = {
    "fixed": {
        "pilot": "implicit_nominal_pilot",
        "slow": "implicit_nominal_sine10_600s",
        "perturbed": "grasp_s7_disturbed120_nominal",
        "sequence": "implicit_nominal_sequence_f0.02_120s",
        "fast": "implicit_nominal_sine_f0.1_120s",
    },
    "full": {
        "pilot": "implicit_reach10_s7_pilot_eval",
        "slow": "implicit_reach10_s7_sine10_600s",
        "perturbed": "implicit_reach10_s7_disturbed120",
        "sequence": "implicit_full_sequence_f0.02_120s",
        "fast": "implicit_full_sine_f0.1_120s",
    },
    "hands": {
        "pilot": "implicit_hands_reach10_s7_pilot_eval",
        "slow": "implicit_hands_reach10_s7_sine10_600s",
        "perturbed": "implicit_hands_reach10_s7_disturbed120",
        "sequence": "implicit_hands_sequence_f0.02_120s",
        "fast": "implicit_hands_sine_f0.1_120s",
    },
}
METRICS = [
    "control", "model", "model_sha256", "scene_sha256", "seed", "task", "target_deg",
    "completed", "early_failure", "termination_reason", "simulated_seconds",
    "rmse_deg", "last_10s_rmse_deg", "max_abs_error_deg", "hold_success", "tracking_pass",
    "both_hands_contact_fraction", "peak_contact_force_N", "peak_robot_collision_N",
    "max_grasp_drift_deg", "max_actual_palm_error_mm", "max_motor_target_gap_rad",
    "max_command_speed_rad_s", "max_command_acceleration_rad_s2", "max_qp_error_mm",
    "qp_failures", "phase_constraint_failures", "reference_clips",
    "max_common_reference_phase_deg", "max_wrist_phase_difference_deg",
    "max_common_residual_deg", "max_differential_residual_deg",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_steps(path):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))["num_timesteps"]


def load_results():
    results, originals = {}, {}
    for controller, protocol in CONTROLLERS.items():
        results[controller], originals[controller] = {}, {}
        for case, directory in protocol.items():
            path = RAW / directory / ("pilot.json" if case == "pilot" else "summary.json")
            if case == "pilot":
                summaries = json.loads(path.read_text())["cases"]
            else:
                summaries = [json.loads(path.read_text())]
            compact = [{k: s[k] for k in METRICS if k in s} for s in summaries]
            for s in compact:
                s["raw_directory"] = str(path.parent.relative_to(ROOT))
            results[controller][case], originals[controller][case] = compact, summaries
    pairs = {}
    for case in CONTROLLERS["fixed"]:
        reference = originals["fixed"][case]
        pairs[case] = all(
            len(reference) == len(originals[c][case]) and all(
                all(a[k] == b[k] for k in ["scene_sha256", "task", "seed", "target_deg", "initial_state"])
                for a, b in zip(reference, originals[c][case]))
            for c in ["full", "hands"])
    if not all(pairs.values()):
        raise ValueError(f"Unmatched experiment settings/initial states: {pairs}")
    return results, pairs


def training_records():
    records = []
    for directory in sorted(RUNS.glob("implicit*_s7_pilot")):
        path = directory / "config.json"
        metadata = json.loads(path.read_text())
        records.append({
            **{k: metadata[k] for k in ["interface_version", "stage", "seed", "control_mode", "action_dim",
                                       "requested_steps", "completed_steps", "cumulative_steps", "resume",
                                       "wall_seconds", "scene_sha256", "source_sha256", "motor_execution"]},
            "directory": str(directory.relative_to(ROOT)), "metadata_sha256": sha(path),
            "selected_best_steps": model_steps(directory / "best/best_model.zip"),
            "best_model_sha256": sha(directory / "best/best_model.zip"),
            "final_model_sha256": sha(directory / "final_model.zip"),
        })
    return records


def curves(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {k: np.array([float(row[k]) for row in rows]) for k in
            ["time", "error_deg", "max_grasp_drift_deg", "peak_force"]}


def residual_ranges():
    path = RAW / CONTROLLERS["full"]["sequence"] / "metrics.csv"
    with path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if float(r["time"]) > 2]
    fields = ["common_residual_deg", "differential_residual_deg", "left_radial_residual_mm",
              "left_axial_residual_mm", "right_radial_residual_mm", "right_axial_residual_mm"]
    return {k: {"min": min(float(r[k]) for r in rows), "max": max(float(r[k]) for r in rows)} for k in fields}


def transition_diagnostics():
    results = {}
    for controller in CONTROLLERS:
        path = RAW / CONTROLLERS[controller]["sequence"] / "metrics.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        peak = max(rows, key=lambda r: abs(float(r["error_deg"])))
        # The current implementation advances physics using the previous goal,
        # then updates the target at the end of each 50 ms environment step.
        held_goals = np.r_[0., [float(r["target_angle"]) for r in rows[:-1]]]
        held_error = np.rad2deg(held_goals - np.array([float(r["wheel_angle"]) for r in rows]))
        first_positive = min(rows, key=lambda r: abs(float(r["time"]) - 5.5))
        results[controller] = {
            "target_update_order": "physics with previous goal, then end-of-step target update",
            "control_period_seconds": .05,
            "held_previous_goal_max_abs_error_deg": float(np.max(np.abs(held_error))),
            "end_of_step_goal_max_abs_error_deg": float(peak["error_deg"]),
            "peak_time_seconds": float(peak["time"]),
            "peak_common_reference_velocity_deg_s": float(peak["common_reference_velocity_deg_s"]),
            "peak_actual_palm_reference_error_mm": float(peak["actual_palm_error_mm"]),
            "first_positive_start_error_deg": float(first_positive["error_deg"]),
            "slow_sine_max_goal_speed_deg_s": 10 * 2 * np.pi * .02,
            "sequence_goal_speed_limit_deg_s": 20,
        }
    return results


def write_readme(output, report):
    results = report["results"]
    names = {"fixed": "固定握持", "hands": "仅手指RL（12维）", "full": "臂手接触RL（18维，主方法）"}
    cases = {"slow": "慢正弦，600s", "perturbed": "随机化慢正弦，120s",
             "sequence": "停留反向，120s", "fast": "较快正弦，120s"}
    lines = [
        "# 共同运动与臂手接触RL：首轮小角度实验", "",
        "本轮完成独立场景、主18维PPO训练和独立12维手指消融。所有控制都使用同一个确定性轮角反馈与连续IK/QP；RL学习接触修正。轮盘只受真实模拟接触驱动。**这是开发阶段小角度试验，尚未完成仿真最终验收或实物迁移。**", "",
        "低误差主要由共同运动控制器提供。主策略有一些降低握持漂移的结果，但伴随更高负载；目前没有证明明显的跟踪优势。停留反向的最大误差超过5°要求，不能由慢正弦的低RMSE替代。", "",
        "## 协议与训练预算", "",
        "从同一个已知预抓握起点开始，目标幅值±10°。慢正弦0.02Hz、较快正弦0.10Hz；停留反向每5秒依次发布0/+10/+10/0/−10/0°端点，目标生成器仍以20°/s限速。三个控制使用相同物理场景、目标、初态、随机化强度和测试种子；汇总脚本逐项检查这些配对条件。", "",
        "训练种子只有7。主方法和手指对照各完成32768步握持及32768步到达训练，训练计算预算各65536步。第二阶段都从各自握持阶段8192步的最佳模型恢复；最终模型累计40960步。评价统一使用各阶段EvalCallback按训练验证回报选出的best模型，没有按下表挑选checkpoint。最佳模型的有效累计步数不同，这不是等有效学习步数的消融。每个模型的选中步数和SHA-256见summary.json。", "",
        "| 模型 | 阶段 | 新增训练步 | 最终累计步 | 最佳模型累计步 |", "|---|---|---:|---:|---:|",
    ]
    for run in report["training"]:
        name = "12维手指" if run["action_dim"] == 12 else "18维臂手"
        lines.append(f"| {name} | {run['stage']} | {run['completed_steps']} | {run['cumulative_steps']} | {run['selected_best_steps']} |")
    lines.extend(["", "## 配对结果", "",
                  "−10°/0°/+10°各60秒保持评价：", "",
                  "| 控制 | −10° RMSE | 0° RMSE | +10° RMSE | 连续10秒保持 |",
                  "|---|---:|---:|---:|---|"])
    for control in ["fixed", "hands", "full"]:
        p = results[control]["pilot"]
        passed = all(r["hold_success"] and r["completed"] for r in p)
        lines.append(f"| {names[control]} | {p[0]['rmse_deg']:.4f}° | {p[1]['rmse_deg']:.4f}° | {p[2]['rmse_deg']:.4f}° | {'3/3通过' if passed else '未全通过'} |")
    lines.extend(["", "动态轨迹全程统计：", "",
                  "| 轨迹 | 控制 | RMSE（°） | 最大误差（°） | 握持漂移代理峰值（°） | 单接触力峰值（N） | 跟踪要求 |",
                  "|---|---|---:|---:|---:|---:|---|"])
    for case, case_name in cases.items():
        for control in ["fixed", "hands", "full"]:
            r = results[control][case][0]
            lines.append(f"| {case_name} | {names[control]} | {r['rmse_deg']:.5f} | {r['max_abs_error_deg']:.4f} | {r['max_grasp_drift_deg']:.4f} | {r['peak_contact_force_N']:.4f} | {'通过' if r['tracking_pass'] else '未通过'} |")
    ranges = report["full_wrist_residual_ranges_after_2s"]["common_residual_deg"]
    lines.extend([
        "", "动态要求为RMSE≤2°、最大误差≤5°、完成请求时长且不提前失败。单接触力不是整只手合力；接触力记录取500Hz物理步内峰值。握持漂移是掌部绕轮轴相位相对轮角的变化，可能包含接触重排、姿态/几何误差，不是指腹切向滑移或力闭合测量。", "",
        "固定参数使用seed1001，随机化样本使用seed1002，停留反向/较快正弦使用seed1003。随机化测试包含摩擦、阻尼、惯量、执行参数、观测噪声与时延，但只有一个参数样本，不能作为统计显著性或鲁棒性证明。", "",
        "停留反向的高误差是起步/运动变化时的短时落后，不是正向准、反向不准：主策略第一次从0°向+10°起步时已有约6.32°误差，从0°向−10°时约6.38°。慢正弦最大目标速度约1.26°/s，而停留后起步的目标速度跳至20°/s；共同参考加速度限制80°/s²，理想情况下从静止加到20°/s也需0.25s，实际臂手运动还有滞后。当前目标在50ms物理执行周期末更新，执行使用上一周期目标，日志比较则使用周期末目标；该时序最多对应1°的目标变化。仅作诊断而与执行时持有的上一目标对齐后，固定/主策略峰值仍约5.47°/5.46°，依然超过5°。正式统计没有改为这个诊断口径。", "",
        f"主策略在停留反向的2秒后，共同相位修正仅在{ranges['min']:.4f}°至{ranges['max']:.4f}°之间变化。当前腕部行为更接近小幅姿态调校；尚不能证明已经学会针对滑移的持续自适应。完整六项腕部修正范围在summary.json中。", "",
        "![配对曲线](pilot_comparison.png)", "",
        "## 执行层问题与修正", "",
        "显式力矩PD的逐物理步速度阻尼在抓握初始化中激发了右腕roll数值振荡。v2改为force-limited affine PD，由implicitfast隐式积分阻尼。刚度、阻尼、重力/偏置前馈及总电机力矩限幅保持一致，没有给轮盘增加驱动，也没有添加无限外力。", "",
        "相同预抓握物理状态，500Hz采样，包含1.2秒抓握初始化及10秒中心保持：", "",
        "| 执行 | 实际臂速峰值 rad/s | 右腕相邻速度相关 | 保持轮角RMSE ° | 最终握持漂移代理 ° | 最大电机力/限幅 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for r in report["execution_validation"]["motor_integration"]:
        lines.append(f"| {r['execution']} | {r['max_arm_velocity_rad_s']:.4f} | {r['right_wrist_adjacent_velocity_correlation']:.4f} | {r['wheel_rmse_deg']:.4f} | {r['max_grasp_drift_deg']:.4f} | {r['max_motor_force_fraction']:.4f} |")
    lines.extend([
        "", "先从已经稳定的隐式抓握状态开始时，显式执行也能稳定。因此结论限于本初始化相关的数值问题，不代表所有显式PD都不稳定。早期显式试训保留为superseded_execution_layer，未恢复到新场景。", "",
        "禁用手轮接触后，手臂参考仍运动，但轮盘角变化严格为0；启用接触的+10°测试轮盘转动8.503°、最终误差−0.02465°。连续IK/QP几何检查通过，物理状态没有由控制器直接覆写。31项原系统/v1/v2测试通过，含actor特权隔离、导出一致性、独立12维训练接口和总电机力矩限幅；课程及全套评价CLI也已检查。", "",
        "![电机执行层500Hz对照](motor_integration.png)", "",
        "## 结论与下一步", "",
        "共同反馈、连续IK与稳定PD执行已经让名义小角度跟踪可靠得多，不能把这些改进算成PPO贡献。18维策略目前主要呈现降低部分握持漂移、增加部分接触负载的取舍；是否需要学习腕部，以及是否优于仅手指，需要更充分的接触挑战和多种子验证。", "",
        "下一步应先定位反向时6°以上误差来自目标速度突变、共同参考限加速度、实际执行滞后还是接触松动，再决定是否修改目标整形或执行层；这些对策本轮未执行。随后用已标定的摩擦/负载/时延范围开展多种子训练，并加入固定学习修正的对照，以区分一次性姿态调校和状态依赖适应。当前力惩罚只在60N以上生效，而本轮单接触峰值约4N，不能期待该项抑制这里的小幅负载增加。阈值应由硬件条件确定。", "",
        "没有完成±30°验证、换手、实物标定或部署。actor不读取模拟接触力，但共同控制器目前仍使用理想轮角；手指联动/驱动单位、轮盘编码器、接触顺应与碰撞过滤需分别验证。未接入视觉、路径规划或CARLA。", "",
        "## 文件与复现", "",
        "- [接口与控制文档](../../docs/METHOD.md)。",
        "- [轻量结果及模型/场景/源码哈希](summary.json)。原始CSV、曲线与日志在outputs/；模型及训练日志在runs/。",
        "- [30秒小角度演示](../../outputs/implicit_full_sequence_demo/rollout.mp4)：确定性共同运动加学习的接触修正。",
        "- 汇总图表可用`python scripts/summarize_results.py`重建，不会重新训练或仿真。最终验收入口见docs/METHOD.md，本轮未运行完整课程与最终suite。",
        "", "上述是2026-10-04实验：当时原驾驶基线核心文件哈希保持不变，未向真实机器人发布命令。2026-10-05目录整理调整了源码路径和导入，场景与权重字节保持不变；原始元数据保留当时的路径及来源哈希。最新隔离迁移见[记录](../isolation.json)和[验证](../isolation_verification.json)。", "",
    ])
    (output / "README.md").write_text("\n".join(lines))


def plots(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"fixed": "Fixed grasp", "hands": "12-action fingers RL", "full": "18-action contact RL"}
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for col, case in enumerate(["perturbed", "sequence"]):
        for controller in ["fixed", "hands", "full"]:
            data = curves(RAW / CONTROLLERS[controller][case] / "metrics.csv")
            for row, metric in enumerate(["error_deg", "max_grasp_drift_deg", "peak_force"]):
                axes[row, col].plot(data["time"], data[metric], label=labels[controller], linewidth=1)
        axes[0, col].set_title("One randomized slow sine" if case == "perturbed" else "Stop and reverse sequence")
        axes[2, col].set_xlabel("Simulation time (s)")
    for row, label in enumerate(["Wheel error (deg)", "Grasp drift proxy (deg)", "Peak single contact (N)"]):
        axes[row, 0].set_ylabel(label)
        for ax in axes[row]:
            ax.grid(alpha=.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Known grasp, +/-10 deg; same deterministic wheel feedback in every controller")
    fig.tight_layout()
    fig.savefig(output / "pilot_comparison.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for execution in ["explicit", "implicit"]:
        path = RAW / "validation_physics500hz" / f"{execution}_500hz.csv"
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        time = np.array([float(r["time"]) for r in rows])
        for ax, metric in zip(axes, ["max_arm_velocity_rad_s", "right_wrist_roll_velocity_rad_s"]):
            ax.plot(time, [float(r[metric]) for r in rows], label=execution, linewidth=.6)
    axes[0].set_ylabel("Max absolute arm speed (rad/s)")
    axes[1].set_ylabel("Right wrist roll speed (rad/s)")
    axes[1].set_xlabel("Time since pregrasp initialization (s)")
    axes[0].set_title("Same PD law and total motor force limits; samples at 500 Hz")
    for ax in axes:
        ax.axvline(1.2, color="black", linestyle="--", linewidth=.8)
        ax.legend()
        ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(output / "motor_integration.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/pilot")
    args = parser.parse_args()
    results, pairs = load_results()
    args.output.mkdir(parents=True, exist_ok=True)
    original_summary = ROOT / "results/pilot/summary.json"
    original_sources = json.loads(original_summary.read_text())["evaluation_source_sha256"]
    sources = [*ROOT.joinpath("src/wheel_manipulation").rglob("*.py"),
               Path(__file__),
               ROOT / "scripts/validate_physics.py", ROOT / "configs/task.yaml"]
    report = {
        "interface_version": "dexsteer-v2-contact-residual", "date": "2026-10-04",
        "scope": "Known initial grasp and geometry; standalone manipulation; deterministic common motion plus bounded contact corrections",
        "accepted_in_simulation": False, "hardware_ready": False, "upstream_connected": False,
        "training_seeds": [7], "test_seeds": [1001, 1002, 1003],
        "purpose": "Development pilot and matched ablation, not final held-out acceptance",
        "matched_settings_and_initial_states": pairs,
        "results": results, "training": training_records(),
        "full_wrist_residual_ranges_after_2s": residual_ranges(),
        "transition_diagnostics": transition_diagnostics(),
        "execution_validation": json.loads((RAW / "validation_physics500hz/validation.json").read_text()),
        "evaluation_source_sha256": original_sources,
        "report_rebuild_source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "limitations": ["One independent training seed and short grasp/reach10 training stages",
                        "Only +/-10 degrees tested in the matched pilot; no hand switching",
                        "Randomization strength 1 evaluated on only one sampled parameter set",
                        "Actor has no contact forces; wheel encoder in deterministic feedback remains ideal",
                        "No hardware calibration or transfer; collision filtering and rigid contact remain approximations",
                        "Low tracking error is largely supplied by deterministic common motion",
                        "No upstream vision, navigation or CARLA integration in v2"],
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plots(args.output)
    write_readme(args.output, report)
    for case in ["slow", "perturbed", "sequence", "fast"]:
        print(case)
        for controller in ["fixed", "hands", "full"]:
            r = results[controller][case][0]
            print(controller, {k: r[k] for k in ["rmse_deg", "max_abs_error_deg", "max_grasp_drift_deg", "peak_contact_force_N", "tracking_pass"]})


if __name__ == "__main__":
    main()
