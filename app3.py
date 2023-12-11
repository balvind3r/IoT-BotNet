import streamlit as st 
import pickle
# import keras
import pandas as pd
from matplotlib import pyplot as plt
# import dangerDetection
dfx = pd.read_csv("TestDx.csv")
dfy = pd.read_csv("TestDy.csv")
plt.style.use("ggplot")


df_master = ["Confusion Matrices", "Model Accuracies","Type Detection"]
col_master = st.sidebar.selectbox("Select a Differentiator: ",df_master)

if(col_master == "Model Accuracies"):
    df = ["ANN", "Random Forest", "RNN", "SVM", "XGBoost"]
    col = st.sidebar.selectbox("Select a model: ",df)
    
    if(col == "ANN"):
        st.markdown("# **ANN**")
        st.markdown(""" The bar graph represents the train-test-validation dataset accuracies.""")
        image_path = "6. Danger XG accuracies.png"
        st.image(image_path, use_column_width=True)

    if(col == "Random Forest"):
        st.markdown("# **Random Forest**")
        st.markdown("""  The bar graph represents the train-test-validation dataset accuracies.""")
        image_path = "15. Danger RF accuracies.png"
        st.image(image_path, use_column_width=True)
        
    if(col == "RNN"):
        st.markdown("# **RNN**")
        st.markdown("""  The bar graph represents the train-test-validation dataset accuracies.""")
        image_path = "11. Danger RNN accuracies.png"
        st.image(image_path, use_column_width=True)
        
    if(col == "SVM"):
        st.markdown("# **SVM**")
        st.markdown(""" The bar graph represents the train-test-validation dataset accuracies.""")
        image_path = "13. Danger SVM accuracies.png"
        st.image(image_path, use_column_width=True)
        
    if(col == "XGBoost"):
        st.markdown("# **XGBoost**")
        st.markdown("""  The bar graph represents the train-test-validation dataset accuracies.""")
        image_path = "6. Danger XG accuracies.png"
        st.image(image_path, use_column_width=True)

if(col_master == "Confusion Matrices"):
    df = ["ANN", "Random Forest", "RNN", "SVM", "XGBoost"]
    col = st.sidebar.selectbox("Select a model: ",df)
    
    if(col == "ANN"):
        st.markdown("# **ANN**")
        st.markdown("""  The confusion matrix provides a summary of the predictions made by a classification model, comparing them to the actual true values. This information is useful in calculation precision and recall of our model. This matrix gives us an idea about how accurate our model is working when trained on a given dataset.""")
        image_path = "9. Danger ANN confusion matrix.png"
        st.image(image_path, use_column_width=True)
        data = {
            'Index': [1,2,3,4],
            'Attributes': ['True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'],
            'Count': [1.5e5, 1.5e5, 23, 50]
        }
        df = pd.DataFrame(data).set_index('Index')

        # Display as a table
        st.table(df)
    
    if(col == "Random Forest"):
        st.markdown("# **Random Forest**")
        st.markdown("""  The confusion matrix provides a summary of the predictions made by a classification model, comparing them to the actual true values. This information is useful in calculation precision and recall of our model. This matrix gives us an idea about how accurate our model is working when trained on a given dataset.""")
        image_path = "16. Danger RF confusion matrix.png"
        st.image(image_path, use_column_width=True)
        data = {
            'Index': [1, 2, 3, 4],
            'Attributes': ['True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'],
            'Count': [1.5e5, 1.5e5, 45, 77]
        }        
        df = pd.DataFrame(data).set_index('Index')

        # Display as a table
        st.table(df)
        
    if(col == "RNN"):
        st.markdown("# **RNN**")
        st.markdown("""  The confusion matrix provides a summary of the predictions made by a classification model, comparing them to the actual true values. This information is useful in calculation precision and recall of our model. This matrix gives us an idea about how accurate our model is working when trained on a given dataset.""")
        image_path = "12. Danger RNN confusion matrix.png"
        st.image(image_path, use_column_width=True)
        data = {
            'Index': [1, 2, 3, 4],
            'Attributes': ['True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'],
            'Count': [1.5e5, 1.5e5, 65, 86]
        }
        df = pd.DataFrame(data).set_index('Index')

        # Display as a table
        st.table(df)
        
    if(col == "SVM"):
        st.markdown("# **SVM**")
        st.markdown("""  The confusion matrix provides a summary of the predictions made by a classification model, comparing them to the actual true values. This information is useful in calculation precision and recall of our model. This matrix gives us an idea about how accurate our model is working when trained on a given dataset.""")
        image_path = "14. Danger SVM confusion matrix.png"
        st.image(image_path, use_column_width=True)
        data = {
            'Index': [1, 2, 3, 4],
            'Attributes': ['True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'],
            'Count': [1.5e5, 1.5e5, 1.4e2, 59]
        }        
        df = pd.DataFrame(data).set_index('Index')

        # Display as a table
        st.table(df)
        
    if(col == "XGBoost"):
        st.markdown("# **XGBoost**")
        st.markdown("""  The confusion matrix provides a summary of the predictions made by a classification model, comparing them to the actual true values. This information is useful in calculation precision and recall of our model. This matrix gives us an idea about how accurate our model is working when trained on a given dataset.""")
        image_path = "5. Danger XG cofusion matrix.png"
        st.image(image_path, use_column_width=True)
        data = {
            'Index': [1, 2, 3, 4],
            'Attributes': ['True Positive (TP)', 'True Negative (TN)', 'False Positive (FP)', 'False Negative (FN)'],
            'Count': [1.5e5, 1.5e5, 6, 46]
        }
        df = pd.DataFrame(data).set_index('Index')

        # Display as a table
        st.table(df)

    
elif(col_master == "Type Detection"):
    st.markdown("# Master Model Outputs for Type Detection")
    st.markdown(""" This page shows the predictions on test data split. The predictions are made on
    randomly generated data subsets. Additionally, they are compared to actual values mentioned in the dataset. """)
    st.subheader("")
   

    predictions={
        # 'Index': [],
        'Values': [7, 5, 7, 9, 8, 4, 5, 0, 9, 7, 7, 5, 0, 1, 10, 9, 5, 5, 4, 0, 3, 0, 7, 8, 8, 5, 8, 7, 5, 0,7, 6, 6, 0, 5, 6, 10, 10, 10, 4, 5, 5, 9, 7, 3, 9, 8, 9, 3, 10, 10, 1, 0, 2, 2, 0, 0, 0, 6, 10,7, 4, 4, 4, 4, 1, 8, 4, 0, 0, 0, 10, 9, 6, 2, 2, 5, 10, 5, 8, 10, 5, 7, 9, 9, 2, 5, 5, 3, 7, 9,6, 9, 9, 1, 0, 7, 10, 6, 0],
        'Prediction': [7, 5, 7, 9, 8, 4, 5, 0, 9, 7, 7, 5, 0, 1, 10, 9, 5, 5, 4, 0, 3, 0, 7, 8, 8, 5, 8, 7, 5, 0,7, 6, 6, 0, 5, 6, 10, 10, 10, 4, 5, 5, 9, 7, 3, 9, 8, 9, 3, 10, 10, 1, 0, 2, 2, 0, 0, 0, 6, 10,7, 4, 4, 4, 4, 1, 8, 4, 0, 0, 0, 10, 9, 6, 2, 2, 5, 10, 5, 8, 10, 5, 7, 9, 9, 2, 5, 5, 3, 7, 9,6, 9, 9, 1, 0, 7, 10, 6, 0]
    }
    st.table(predictions)

