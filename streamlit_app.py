{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d87104d-c150-40f1-993a-cff52ba2abce",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import streamlit as st\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "from PIL import Image\n",
    "\n",
    "# Load your Keras model\n",
    "try:\n",
    "    model = tf.keras.models.load_model(r\"summer_project_model1.keras\")\n",
    "except:\n",
    "    st.error(\"Failed to load model. Please check the model path.\")\n",
    "    st.stop()\n",
    "    \n",
    "# Define class names if applicable (replace with your actual class names)\n",
    "class_names = [\"Potato___Early_blight\", \"Potato___healthy\", \"Potato___Late_blight\"]\n",
    "\n",
    "# Streamlit app title\n",
    "st.title('Deep Learning Model Deployment with Streamlit')\n",
    "# Upload image through file uploader\n",
    "uploaded_file = st.file_uploader(\"Choose an image...\", type=\"jpg\")\n",
    "if uploaded_file is not None:\n",
    "    if uploaded_file.type != 'image/jpeg':\n",
    "        st.warning('Please upload a JPG image.')\n",
    "    else:\n",
    "    # Read the image file and preprocess it\n",
    "        image = Image.open(uploaded_file)\n",
    "        image = image.resize((256, 256))  # Resize image to match model's expected sizing\n",
    "        image = np.array(image)  # Convert to numpy array\n",
    "        image = np.expand_dims(image, axis=0)  # Add batch dimension\n",
    "        \n",
    "        # Display uploaded image\n",
    "        st.image(image[0], caption='Uploaded Image.', use_column_width=True)\n",
    "        \n",
    "        # Classify the image\n",
    "        st.write(\"Classifying...\")\n",
    "        predictions =model.predict(tf.convert_to_tensor(image))\n",
    "        predicted_class = class_names[np.argmax(predictions)]\n",
    "        \n",
    "        # Display prediction\n",
    "        st.write(f'Prediction: {predicted_class}')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
