# 2026 APMCM Problem C: Acoustic Communication and Playback Responses of African Penguins at Sea

2026 年亚太地区大学生数学建模竞赛（APMCM）C 题两人协作仓库。

## 目录结构

```text
2026-APMCM-Problem-C/
├── data/                  # 本地数据目录（可选放置 CSV；默认自动读取上级赛题 date/ 目录）
├── outputs/
│   ├── figures/           # 自动生成的高清图表
│   └── tables/            # 自动生成的统计与模型结果表
├── paper/                 # LaTeX 论文源码与排版文件
├── src/
│   ├── config.py          # 路径、随机种子与绘图规范配置
│   ├── q1_acoustics_classification.py  # 问题一：声学特征描述、非参数检验与分类器
│   └── q2_signal_degradation.py        # 问题二：保真度融合、衰减拟合与有效通信距离
├── .gitignore
├── requirements.txt
└── README.md
```

## 两人协作分工

- **成员 A（建模与代码主笔）**：负责 `src/` 下 Q1–Q4 模型实现、参数拟合、Bootstrap 不确定性量化与情景仿真。
- **成员 B（统计、图表与论文主笔）**：负责统计检验校验、图表美化、`paper/` 下 LaTeX 撰写与 Summary Sheet 精修。

## 快速开始

```bash
pip install -r requirements.txt
python src/q1_acoustics_classification.py
python src/q2_signal_degradation.py
```

如需指定数据目录，可设置环境变量 `APMCM_C_DATA_DIR`。
