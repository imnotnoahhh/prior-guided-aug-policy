运行环境

系统Ubuntu24.04 64位
cuda12.8
nvidia驱动 570.113.20

和baseline（不会保存文件但会保留日志）
python run_baseline.py | tee logs/baseline_result.log


跑A阶段（保留文件和日志）
python main_phase_a.py > logs/phase_a_full.log 2>&1 &
