import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 加载模型
try:
    model = joblib.load('random_forest_model.pkl')
except FileNotFoundError:
    st.error("Model file 'random_forest_model.pkl' not found. Please upload the model file.")
    st.stop()

# 特征范围定义
feature_names = [
    "Gender", "BMI", "Race"
]
feature_ranges = {
    "BMI": {"type": "numerical", "min": 25, "max": 100, "default": 25},
    "Gender": {"type": "categorical", "options": ["1", "2"]},
    "Race": {"type": "categorical", "options": ["1", "2", "3", "4", "5"]},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")
st.header("Enter the following feature values:")

feature_values = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        feature_values[feature] = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        feature_values[feature] = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )

# 处理分类特征
label_encoders = {}
for feature, properties in feature_ranges.items():
    if properties["type"] == "categorical":
        label_encoders[feature] = LabelEncoder()
        label_encoders[feature].fit(properties["options"])
        feature_values[feature] = label_encoders[feature].transform([feature_values[feature]])[0]

# 转换为模型输入格式
features = pd.DataFrame([feature_values], columns=feature_names)

# 预测与 SHAP 可视化
if st.button("Predict"):
    try:
        # ... [之前的预测代码保持不变] ...

        # 计算 SHAP 值
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)

        # 调试输出维度
        st.write("SHAP Values Shape:", np.array(shap_values).shape)
        st.write("Features Shape:", features.shape)

        # 动态选择预测类别的 SHAP 值
        class_index = predicted_class
        if isinstance(shap_values, list):
            # 分类模型：shap_values 是列表，每个元素对应一个类别
            shap_value_for_plot = shap_values[class_index][0, :]
            base_value = explainer.expected_value[class_index]
        else:
            # 回归模型：shap_values 是单一数组
            shap_value_for_plot = shap_values[0, :]
            base_value = explainer.expected_value

        # 创建新的 Matplotlib 画布
        plt.figure(figsize=(10, 4))

        # 绘制 SHAP 力图
        shap.force_plot(
            base_value,
            shap_value_for_plot,
            features.iloc[0].values,
            feature_names=features.columns,
            matplotlib=True,
            show=False
        )
        plt.title(f"SHAP Analysis: AKI Probability {probability:.2f}%", fontsize=12)
        plt.tight_layout()
        plt.savefig("shap_force_plot.png", bbox_inches="tight", dpi=300)
        plt.close()
        st.image("shap_force_plot.png", caption="SHAP Feature Impact", use_column_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
