import streamlit as st
import numpy as np
import pandas as pd
import joblib
from streamlit_echarts import st_echarts

# 加载模型和特征名称
def load_model(model_select):
    if model_select == 'SVMmodel':
        svm_model = joblib.load('SVM_Pyrite_Classifier.pkl')
        return svm_model
    elif model_select == 'RFmodel':
        rf_model = joblib.load('RF_Pyrite_Classifier.pkl')
        return rf_model

feature_names = ['Co', 'Ni', 'Cu', 'Zn', 'As', 'Se', 'Ag', 'Sb', 'Te', 'Au', 'Pb', 'Bi']
prediction_labels = {
    0: 'Skarn',
    1: 'VMS',
    2: 'Epithermal',
    3: 'Orogenic',
    4: 'Carlin',
    5: 'Porphyry',
    6: 'Magmatic Sulfide'
}

def predict(features, model_select):
    model = load_model(model_select)
    prediction = model.predict(features.reshape(1, -1))
    prediction_label = prediction_labels[prediction[0]]
    probabilities = model.predict_proba(features.reshape(1, -1))[0]
    return prediction_label, probabilities

def main():
    # 页面标题
    st.title('Prediction Model for Se-Te Ore Deposits Based on Pyrite Trace Elements')
    st.sidebar.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    
    # 左侧栏 - 模型选择
    st.sidebar.title('Model Selection')
    model_select = st.sidebar.selectbox(
        'Choose Model',
        ('SVMmodel', 'RFmodel')
    )

    # 左侧栏 - 数据输入
    st.sidebar.title('Data Input')
    option = st.sidebar.radio(
        'Choose Data Input Method:',
        ['Manually Enter Features', 'Upload Data Table']
    )
    st.sidebar.info("Preliminary Prediction!")

    features = None  # 初始化 features 变量

    if option == 'Manually Enter Features':
        st.title('Manually Enter Features')
        feature_inputs = []
        for i in range(len(feature_names)):
            feature = st.number_input(f'Enter value for {feature_names[i]}', value=0.00001)
            feature_inputs.append(feature)
        features = np.array(feature_inputs).reshape(1, -1)
        if np.any(features <= 0):
            st.warning('Input data must be positive numerical values. Please enter valid feature values.')
            return
    else:
        st.title('Upload Data Table')
        st.subheader('Example Data Table Style:')
        example_data = {
            'Co': [1.2, 0.8, 1.5],
            'Ni': [0.3, 0.2, 0.4],
            'Cu': [2.1, 1.8, 2.2],
            'Zn': [2.1, 1.8, 2.2],
            'As': [2.1, 1.8, 2.2],
            'Se': [2.1, 1.8, 2.2],
            'Ag': [2.1, 1.8, 2.2],
            'Sb': [2.1, 1.8, 2.2],
            'Te': [2.1, 1.8, 2.2],
            'Au': [2.1, 1.8, 2.2],
            'Pb': [2.1, 1.8, 2.2],
            'Bi': [2.1, 1.8, 2.2]
        }
        example_df = pd.DataFrame(example_data)
        st.table(example_df)

        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                features = data[feature_names]
                if np.any(features <= 0):
                    st.warning('Input data must be positive numerical values. Please enter valid feature values.')
                    return
                st.subheader('Uploaded Data Table Example')
                st.table(data.head(3))
            except Exception as e:
                st.error('Error reading file. Please check the uploaded data table file.')
                st.error(str(e))

    # 预测按钮
    if st.button('Predict'):
        if features is not None:
            prediction, probabilities = predict(features, model_select)
            st.subheader('Prediction Result')
            st.write(f'Prediction: {prediction}')
            st.write('Prediction Probabilities:')
            for label, prob in zip(prediction_labels.values(), probabilities):
                st.write(f'{label}: {prob}')

            # Echarts pie chart
            chart_data = [{'value': float(prob), 'name': label} for label, prob in zip(prediction_labels.values(), probabilities)]
            opts = {"title": {"text": "Prediction Probabilities", "textStyle": {"fontSize": 24}}, 
                    "series": [{"type": "pie", "data": chart_data, 
                                "label": {"show": True, "formatter": "{b}: {d}%", "fontSize": 15}}]}
            st.write(st_echarts(options=opts, width="700px", height="500px"))
        else:
            st.warning('Please upload a data table or manually enter features.')

if __name__ == '__main__':
    main()

