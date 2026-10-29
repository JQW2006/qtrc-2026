# 2026 量化交易研究大赛 · 私榜第一名方案

私榜得分 36.2404（×10⁻⁴，加权零均值 R²）。策略由 CatBoost、XGBoost、D1 冻结、D1 在线四路加权融合。

**在线报告**：<https://jqw2006.github.io/qtrc-2026/>

| 目录 | 内容 |
|---|---|
| `report/` | 策略报告（总览、time_id、本地验证、GBDT、MLP、工程实现、QA） |
| `inference/` | 提交的推理包，只依赖 numpy |
| `train/` | 复现训练代码 |
| `docs/` | 赛题文档 |

报告网页由 `.github/workflows/pages.yml` 在每次修改 `report/` 后自动构建发布。
