import streamlit as st
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# load the ANN model from file
model = tf.keras.models.load_model('ANN_Iklim.h5')

# Define scaler as a global variable
scaler = MinMaxScaler()


# Function to preprocess input data
def preprocess_input_data(input_data):
    preprocessed_data = scaler.fit_transform(input_data)
    return preprocessed_data


# Function to make prediction
def predict(model, preprocessed_data):
    predictions = model.predict(preprocessed_data)
    return predictions


# Main function
def main():
    # Set the title and description of your web app
    st.title("ANN Prediction Curah Hujan Kota Jakarta")
    st.write("Enter the input data to get predictions.")

    # Get user input
    Suhu_Minimum = st.number_input("Suhu Minimum", value=0.00)
    Suhu_Maksimum = st.number_input("Suhu Maksimum", value=0.00)
    Suhu_Rata2 = st.number_input("Suhu Rata-Rata", value=0.00)
    Kelembapan_Rata2 = st.number_input("Kelembapan Rata-Rata", value=0.00)
    Lama_Penyinaran_Matahari = st.number_input(
        "Lama Penyinaran Matahari", value=0.00)
    Kecepatan_Angin_Maksimum = st.number_input(
        "Kecepatan Angin Maksimum", value=0.00)
    Arah_Angin_Saat_Kecepatan_Maksimum = st.number_input(
        "Arah Angin Saat Kecepatan Maksimum", value=0.00)
    Kecepatan_Angin_Rata2 = st.number_input(
        "Kecepatan Angin Rata-Rata", value=0.00)
    Arah_Angin_Terbanyak = st.number_input("Arah Angin Terbanyak", value=0.00)

    # Preprocess the input data
    input_data = [[
        Suhu_Minimum,
        Suhu_Maksimum,
        Suhu_Rata2,
        Kelembapan_Rata2,
        Lama_Penyinaran_Matahari,
        Kecepatan_Angin_Maksimum,
        Arah_Angin_Saat_Kecepatan_Maksimum,
        Kecepatan_Angin_Rata2,
        Arah_Angin_Terbanyak]]
    preprocessed_data = preprocess_input_data(input_data)

    # Make predictions
    if st.button("Predict"):
        predictions = predict(model, preprocessed_data)

    # Menggunakan argmax untuk mendapatkan hasil prediksi tunggal
        predicted_category_index = tf.argmax(predictions, axis=1).numpy()
        predicted_category = [
            "Berawan", "Hujan Ringan", "Hujan Sedang", "Hujan Lebat",
            "Hujan Sangat Lebat", "Hujan Ekstrem"]

    # Menampilkan hasil prediksi tunggal
        predicted_result = predicted_category[predicted_category_index[0]]
        st.write("Prediction:", predicted_result)


# Run the app
if __name__ == '__main__':
    main()
