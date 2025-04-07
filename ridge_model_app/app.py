import streamlit as st
import pandas as pd
import numpy as np
from model_preprocessing import load_model, preprocess_data
from model_preprocessing import custom_tokenizer  # Make sure this line is there


def main():
    st.title('Ridge Model Application')
    st.write('Upload your dataset and get predictions from your Ridge model.')

    uploaded_file = st.file_uploader('Upload TSV or CSV File', type=['csv', 'tsv'])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.tsv'):
            df = pd.read_csv(uploaded_file, delimiter='\t')  # Read TSV file
        else:
            df = pd.read_csv(uploaded_file)  # Read CSV file

        st.write('Uploaded Data:')
        st.write(df.head())

        # Check if required columns are present
        required_columns = ['name', 'item_description', 'brand_name', 'category_name']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
            return

        model = load_model()
        processed_data = preprocess_data(df)

        predictions = model.predict(processed_data)
        df['Predictions'] = np.expm1(predictions)  # Convert predictions back to original scale

        st.write('Predictions:')
        st.write(df)

        # Option to download predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label='Download Predictions as CSV',
            data=csv,
            file_name='predictions.csv',
            mime='text/csv',
        )

if __name__ == '__main__':
    main()

