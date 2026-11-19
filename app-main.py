import streamlit as st
import pandas as pd
import joblib

# 页面显示用的因子名称（与上传 Excel 的列名一致）
display_cols = ['Age', 'CA125', 'CA15-3', 'CYFRA21-1', 'NSE', 'ALB', 'Urea']
# 模型训练时使用的特征名称（顺序与 display_cols 一一对应）
model_cols = ['Age', 'CA125', 'CA153', 'CYFRA211', 'NSE', 'ALB', 'Urea']

# 初始化 session_state 中的 data
if 'data' not in st.session_state:
    st.session_state['data'] = pd.DataFrame(columns=display_cols + ['Prediction', 'Label'])

# 设置页面为宽模式
st.set_page_config(layout="wide")
st.sidebar.image("hospital_logo2.png", caption="", width=300)
# Language setting
lang = st.sidebar.selectbox('Choose language', ['中文', 'English'], index=1)

# Footer
if lang == '中文':
    st.sidebar.subheader('程序说明')
    st.sidebar.write(
        "<p style='font-size: 12px;'>申明: 这款小程序旨在提供一般信息，不能替代专业医疗建议或诊断。如果您对自己的健康有任何担忧，请务必咨询合格的医疗保健专业人员。</p>",
        unsafe_allow_html=True)
else:
    st.sidebar.subheader('Program Description')
    st.sidebar.write(
        "<p style='font-size: 12px;'>Disclaimer: This mini app is designed to provide general information and is not a substitute for professional medical advice or diagnosis. Always consult with a qualified healthcare professional if you have any concerns about your health.</p>",
        unsafe_allow_html=True)

st.image("hospital_logo.png", caption="")
if lang == '中文':
    st.header("结节风险评估模型（NRA）V1.0 ")
else:
    st.header("NoduleRisk Assessment（NRA) nodel V1.0")


def number_with_unit(label_zh, label_en, unit, min_value, max_value, value, **kwargs):
    """带单位的数字输入框：左侧输入，右侧显示单位"""
    input_col, unit_col = st.columns([8, 2])
    v = input_col.number_input(label_zh if lang == '中文' else label_en,
                               min_value=min_value, max_value=max_value, value=value, **kwargs)
    unit_col.markdown(unit)
    return v


# 将输入分成三列
col1, col2, col3 = st.columns(3)

with col1:
    age = number_with_unit("年龄", "Age", "岁" if lang == '中文' else "years", 0, 120, 62)
    ca125 = number_with_unit("糖类抗原125", "CA125", "U/mL", 0.00, 10000.00, 16.00)
    ca153 = number_with_unit("癌抗原15-3", "CA15-3", "U/mL", 0.00, 2000.00, 12.55)

with col2:
    cyfra = number_with_unit("细胞角蛋白19片段", "CYFRA21-1", "ng/mL", 0.00, 200.00, 2.73)
    nse = number_with_unit("神经元特异性烯醇化酶", "NSE", "ng/mL", 0.00, 500.00, 16.40)

with col3:
    alb = number_with_unit("白蛋白", "ALB", "g/L", 0.0, 100.0, 39.5, step=0.1, format="%.1f")
    urea = number_with_unit("尿素", "Urea", "mmol/L", 0.00, 100.00, 5.26)

# Unpickle classifier
mm = joblib.load('logistic_regression.pkl')

# If button is pressed
if lang == '中文':
    submit_button = st.button("提交")
else:
    submit_button = st.button("Submit")

if submit_button:
    values = [age, ca125, ca153, cyfra, nse, alb, urea]
    X = pd.DataFrame([values], columns=model_cols)

    # Get prediction
    result_prob_pos = mm.predict_proba(X)[0][1] * 100

    if lang == '中文':
        st.text(f"肺结节恶性概率是: {round(result_prob_pos, 2)}%")
    else:
        st.text(f"The probability of malignancy is: {round(result_prob_pos, 2)}%")

    # 将输入和预测结果添加到表格中
    new_data = pd.DataFrame([values + [result_prob_pos / 100, None]],
                            columns=st.session_state['data'].columns)
    st.session_state['data'] = pd.concat([st.session_state['data'], new_data], ignore_index=True)

# 上传文件按钮
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # 兼容两种列名：页面显示名（CA15-3、CYFRA21-1）和模型特征名（CA153、CYFRA211）
    column_mapping = {'CA153': 'CA15-3', 'CYFRA211': 'CYFRA21-1', 'Nse': 'NSE'}
    df = df.rename(columns=column_mapping)

    # 'label' 列（如果有）不参与计算，只用于对照显示
    label_column = 'label'

    missing_cols = [col for col in display_cols if col not in df.columns]

    if missing_cols:
        st.error(f"Missing columns in the uploaded file: {', '.join(missing_cols)}")
    else:
        X = df[display_cols].copy()
        X.columns = model_cols
        result = df[display_cols].copy()
        result['Prediction'] = mm.predict_proba(X)[:, 1]
        result['Label'] = df[label_column] if label_column in df.columns else None
        st.session_state['data'] = pd.concat([st.session_state['data'], result], ignore_index=True)

st.write(st.session_state['data'])

st.markdown('<div style="font-size: 12px; text-align: right;">郑州大学第一附属医院</div>', unsafe_allow_html=True)
st.markdown(
    '<div style="font-size: 12px; text-align: right;">THE FIRST AFFILIATED HOSPITAL OF ZHENGZHOU UNIVERSITY</div>',
    unsafe_allow_html=True)
