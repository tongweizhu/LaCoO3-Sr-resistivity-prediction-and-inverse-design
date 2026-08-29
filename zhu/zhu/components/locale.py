"""Small, session-local language layer for the desktop workbench.

English is deliberately the default so a fresh GitHub checkout opens with an
international scientific vocabulary.  Chinese remains available from the
header selector.  The page modules may continue to keep their validated
Chinese source strings; this module translates the user-facing vocabulary at
render time without changing API field names or file formats.
"""

from __future__ import annotations

from typing import Any


DEFAULT_LANGUAGE = "en"
LANGUAGE_OPTIONS = {"en": "English", "zh": "中文"}


_TRANSLATIONS: dict[str, dict[str, str]] = {
    "单次预测": {"en": "Single Prediction", "zh": "单次预测"},
    "批量预测": {"en": "Batch Prediction", "zh": "批量预测"},
    "逆向设计": {"en": "Inverse Design", "zh": "逆向设计"},
    "模型微调": {"en": "Model Fine-tuning", "zh": "模型微调"},
    "LSCO 电阻率预测工作台": {"en": "LSCO Resistivity Prediction Workbench", "zh": "LSCO 电阻率预测工作台"},
    "单屏参数工作台 · 训练范围检查 · 预测曲线 · 浏览器本地下载": {
        "en": "Single-screen inputs · training-range checks · prediction curve · browser downloads",
        "zh": "单屏参数工作台 · 训练范围检查 · 预测曲线 · 浏览器本地下载",
    },
    "模型输入": {"en": "Model Inputs", "zh": "模型输入"},
    "电阻率—温度曲线": {"en": "Resistivity–Temperature Curve", "zh": "电阻率—温度曲线"},
    "温度扫描": {"en": "Temperature Scan", "zh": "温度扫描"},
    "最低温度（K）": {"en": "Minimum temperature (K)", "zh": "最低温度（K）"},
    "最高温度（K）": {"en": "Maximum temperature (K)", "zh": "最高温度（K）"},
    "生成预测曲线": {"en": "Generate prediction curve", "zh": "生成预测曲线"},
    "下载 CSV": {"en": "Download CSV", "zh": "下载 CSV"},
    "下载 PNG": {"en": "Download PNG", "zh": "下载 PNG"},
    "下载 XLSX": {"en": "Download XLSX", "zh": "下载 XLSX"},
    "下载结果": {"en": "Download result", "zh": "下载结果"},
    "当前没有可下载的结果。": {"en": "There is no downloadable result yet.", "zh": "当前没有可下载的结果。"},
    "下载失败：": {"en": "Download failed: ", "zh": "下载失败："},
    "上传文件": {"en": "Upload file", "zh": "上传文件"},
    "点击或拖拽上传": {"en": "Click or drop to upload", "zh": "点击或拖拽上传"},
    "暂无可预览的数据。": {"en": "No preview data yet.", "zh": "暂无可预览的数据。"},
    "单位待确认": {"en": "unit pending confirmation", "zh": "单位待确认"},
    "LSCO LAB": {"en": "LSCO LAB", "zh": "LSCO LAB"},
    "RESISTIVITY WORKBENCH": {"en": "RESISTIVITY WORKBENCH", "zh": "RESISTIVITY WORKBENCH"},
    "当前模型": {"en": "Current model", "zh": "当前模型"},
    "当前加载模型": {"en": "Currently loaded model", "zh": "当前加载模型"},
    "模型范围内": {"en": "Within model range", "zh": "模型范围内"},
    "外推结果": {"en": "Extrapolated result", "zh": "外推结果"},
    "等待输入": {"en": "Waiting for input", "zh": "等待输入"},
    "预测序列": {"en": "Prediction series", "zh": "预测序列"},
    "最小电阻率": {"en": "Minimum resistivity", "zh": "最小电阻率"},
    "最大电阻率": {"en": "Maximum resistivity", "zh": "最大电阻率"},
    "输入尺度与范围说明": {"en": "Input scale and range notes", "zh": "输入尺度与范围说明"},
    "重新读取模型信息": {"en": "Reload model metadata", "zh": "重新读取模型信息"},
    "全部字段必填；字段说明与训练范围在悬停提示中。": {
        "en": "All fields are required; hover a field for its definition and training range.",
        "zh": "全部字段必填；字段说明与训练范围在悬停提示中。",
    },
    "Language": {"en": "Language", "zh": "语言"},
    "LSCO Resistivity Prediction": {"en": "LSCO Resistivity Prediction", "zh": "LSCO 电阻率预测"},
    "One application, four workflows: single prediction / batch prediction / inverse design / model fine-tuning": {
        "en": "One application, four workflows: single prediction / batch prediction / inverse design / model fine-tuning",
        "zh": "一个程序，四项流程：单次预测 / 批量预测 / 逆向设计 / 模型微调",
    },
    "目标电阻率": {"en": "Target Resistivity", "zh": "目标电阻率"},
    "候选条件": {"en": "Candidate Conditions", "zh": "候选条件"},
    "开始逆向搜索": {"en": "Start Inverse Search", "zh": "开始逆向搜索"},
    "下载配置": {"en": "Download Configuration", "zh": "下载配置"},
    "非唯一逆解": {"en": "Non-unique Inverse Solution", "zh": "非唯一逆解"},
    "Substrate 仅用于推导 Mismatch": {
        "en": "Substrate is used only to derive Mismatch",
        "zh": "Substrate 仅用于推导 Mismatch",
    },
    "English": {"en": "English", "zh": "英语"},
    "中文": {"en": "Chinese", "zh": "中文"},
    "此项为必填项": {"en": "This field is required.", "zh": "此项为必填项"},
    "请选择允许的类别": {"en": "Select an allowed category.", "zh": "请选择允许的类别"},
    "请输入有限数字": {"en": "Enter a finite number.", "zh": "请输入有限数字"},
    "请输入大于 0 的数值": {"en": "Enter a value greater than 0.", "zh": "请输入大于 0 的数值"},
    "仅允许选择 0 或 1": {"en": "Only 0 or 1 is allowed.", "zh": "仅允许选择 0 或 1"},
    "Model ready. Default test data loaded; generate the prediction curve.": {
        "en": "Model ready. Default test data loaded; generate the prediction curve.",
        "zh": "模型已就绪。默认测试数据已载入，可直接生成预测曲线。",
    },
    "Model ready. Fill every input field, then generate the prediction curve.": {
        "en": "Model ready. Fill every input field, then generate the prediction curve.",
        "zh": "模型已就绪。请填写全部输入字段后生成预测曲线。",
    },
    "温度范围必须是数字": {"en": "Temperature bounds must be numeric.", "zh": "温度范围必须是数字"},
    "温度范围必须是有限数字": {"en": "Temperature bounds must be finite.", "zh": "温度范围必须是有限数字"},
    "最高温度必须大于最低温度": {"en": "Maximum temperature must exceed minimum temperature.", "zh": "最高温度必须大于最低温度"},
}


def language_for(component: Any | None = None) -> str:
    """Return the current session language, falling back to English."""

    session = getattr(component, "session", component)
    language = getattr(session, "_lsco_language", DEFAULT_LANGUAGE)
    return language if language in LANGUAGE_OPTIONS else DEFAULT_LANGUAGE


def set_language(session: Any, language: str) -> None:
    """Store a language choice on the current Rio session only."""

    setattr(session, "_lsco_language", language if language in LANGUAGE_OPTIONS else DEFAULT_LANGUAGE)


def translate(text: str, component: Any | None = None) -> str:
    """Translate a known UI string while leaving variables and API names intact."""

    value = str(text)
    language = language_for(component)
    direct = _TRANSLATIONS.get(value)
    if direct and language in direct:
        return direct[language]
    if language == "zh":
        for entry in _TRANSLATIONS.values():
            if entry.get("en") == value:
                return entry.get("zh", value)
    return value


__all__ = ["DEFAULT_LANGUAGE", "LANGUAGE_OPTIONS", "language_for", "set_language", "translate"]
