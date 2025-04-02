# 🌱 Plant Seedling Classification using Convolutional Neural Networks (CNNs)

## 📌 Context

The agriculture industry, which remains heavily reliant on manual labor, faces challenges in tasks such as sorting and identifying plant seedlings. Despite advances in technology, workers still spend considerable time and effort identifying different plant species and weeds. By leveraging Artificial Intelligence and Deep Learning, this project aims to modernize the agricultural industry, reducing manual labor, increasing crop yields, and improving overall sustainability. The primary goal is to build an AI-driven system that efficiently classifies plant seedlings based on images, helping agricultural workers save time and make better decisions.

## 🎯 Objective

The main objective of this project is to develop a Convolutional Neural Network (CNN) capable of classifying plant seedlings into 12 distinct species using image data.

## 📊 Data Dictionary

The dataset used in this project was created by the Aarhus University Signal Processing group, in collaboration with the University of Southern Denmark. The dataset includes images of plants belonging to 12 different species.

### Plant Species to Classify:
- Black-grass
- Charlock
- Cleavers
- Common Chickweed
- Common Wheat
- Fat Hen
- Loose Silky-bent
- Maize
- Scentless Mayweed
- Shepherds Purse
- Small-flowered Cranesbill
- Sugar beet

### Data Preprocessing & Handling:
- Imbalance in Data: Some species, such as 'Loose Silky-bent' and 'Common Chickweed,' are more abundant than others, leading to class imbalance.
- Image Size Reduction: The original image size was reduced from 128px to 64px to optimize for computational efficiency during training.

## 🧑‍💻 Libraries & Frameworks Used
- TensorFlow & Keras for deep learning model building
- OpenCV for image processing
- Scikit-learn for evaluation and data splitting
- Pandas & NumPy for data manipulation
- Matplotlib & Seaborn for data visualization

### Required Libraries:
- pip install tensorflow==2.15.0 scikit-learn==1.2.2 seaborn==0.13.1 matplotlib==3.7.1 numpy==1.25.2 pandas==1.5.3 opencv-python==4.8.0.76

## 🛠 Model Development & Techniques

### 1️⃣ Initial Model Development
- Optimizers:
  - Initially, I experimented with the SGD optimizer, followed by the Adam optimizer for better convergence and performance.
  - I also adjusted the learning rate to 0.0001 to ensure more stable training.
- Model Architecture:
  - I began with basic Convolutional Layers with different kernel sizes and explored MaxPooling, Dropout, and BatchNormalization to prevent overfitting and improve generalization.
  - The activation function used was ReLU for hidden layers, ensuring non-linearity and faster convergence.

### 2️⃣ Advanced Techniques:
- Data Augmentation:
  - Rotation, Zooming, Shifting were applied to generate variations in the training data, reducing overfitting and improving generalization.
- Transfer Learning:
  - I used VGG16 pre-trained on ImageNet weights to build a transfer learning model.
  - The weights of the VGG16 model were frozen, and only the dense layers were fine-tuned, significantly reducing training time and improving performance.
  - The pre-trained VGG16 model had around 15 million parameters.

### 3️⃣ Evaluation Metrics:
The following metrics were used to evaluate model performance:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
    
These metrics allowed us to assess the model's ability to classify plant seedlings accurately while ensuring the balance between false positives and false negatives.

## 📊 Model Evaluation & Comparison:

### Key Insights:

- Model 6 and Model 7 showed signs of overfitting, so they were discarded.
- Model 9 demonstrated acceptable accuracy but had low performance in terms of recall, making it unsuitable for finalization.
- Model 1 and Model 2 also performed poorly.
- Model 4 emerged as the best performing model, offering a good balance of accuracy, recall, and precision without significant overfitting.

### Final Model Selection:

- Model 4 was selected as the final model due to its reliable performance and ability to generalize well to unseen data, while avoiding overfitting.

## 🌱 Practical Impact & Use Case:

This model can significantly aid in modernizing the agricultural sector by automating the identification of plant seedlings, saving both time and manual labor. With better classification, farmers can enhance their efficiency, monitor plant health, and ultimately achieve higher crop yields. The model can be further expanded to support real-time applications in farm management systems, optimizing agricultural practices.
