# Obesity Level Prediction Using Neural Networks

This project is a machine learning pipeline to predict obesity levels based on various features such as age, gender, physical activity, and food consumption. The project includes steps for data preprocessing, exploratory data analysis (EDA), model building, training, evaluation, and discussion of potential improvements.

## Project Structure

### 1. Dataset:

* The dataset contains 2111 records with 17 columns, including features such as Age, Gender, Height, Weight, and NObeyesdad (target variable representing obesity levels).
* Source: Synthetic obesity dataset.

### 2. Steps in the Pipeline:

**Step 1: Check the Structure of the Dataset**

* Loaded the dataset using pandas.
* Checked for missing values, duplicate rows, and data types.
* Removed 24 duplicate rows.

**Step 2: Exploratory Data Analysis (EDA)**

* Conducted visualizations to analyze:
* Distribution of obesity levels.
* Relationship between age and obesity levels.
* Frequency of physical activity and its impact on obesity levels.
* Gender-based differences in obesity.
* Impact of food and water consumption on obesity levels.

**Step 3: Data Preprocessing**

* One-hot encoded categorical features like Gender, CALC, and MTRANS.
* Normalized numerical features using Min-Max scaling.
* Split the dataset into training (80%) and testing (20%) sets.

**Step 4: Build the Neural Network**

* Created a sequential neural network using TensorFlow/Keras with the following architecture:
* Input Layer: 128 neurons, ReLU activation.
* Hidden Layers: Two layers with 64 and 32 neurons, ReLU activation.
* Output Layer: Number of classes (softmax activation).
* Compiled the model using the Adam optimizer, sparse categorical crossentropy loss, and accuracy as the evaluation metric.

**Step 5: Train the Neural Network**

* Trained the model for 30 epochs with a batch size of 32.
* Monitored training and validation accuracy/loss to identify overfitting or underfitting.

**Step 6: Evaluate the Model**

* Evaluated the model on the test set.
* Achieved test accuracy of X%.
* Visualized the confusion matrix and classification report for detailed insights.

**Step 7: Discussion and Improvements**

* Key findings and areas for improvement:
* Addressed class imbalance with potential use of SMOTE for oversampling.
* Suggested hyperparameter tuning with Grid Search.
* Proposed dropout layers and batch normalization to improve generalization.


## Technologies Used

* Python (Pandas, NumPy, Matplotlib, Seaborn, TensorFlow/Keras, Scikit-learn)
* Jupyter Notebook for experimentation


## How to Use

**1. Clone the Repository:**
```
git clone https://github.com/your-username/obesity-prediction
cd obesity-prediction
```
**2. Install Dependencies:**
```
pip install -r requirements.txt
```
**3. Run the Code:**

* Open the Jupyter notebook or Python script.
* Follow the step-by-step pipeline for loading the dataset, preprocessing, training, and evaluation.

**4. Dataset:**

* Ensure the dataset file (`ObesityDataSet_raw_and_data_sinthetic.csv`) is in the project directory.
  

## Results

* The model successfully predicts obesity levels with a reasonable accuracy.
* The confusion matrix and classification report highlight strengths and weaknesses in predictions for specific classes.


## Future Enhancements

* Perform advanced hyperparameter tuning (e.g., Grid Search).
* Use techniques like SMOTE for handling class imbalance.
* Add dropout layers or batch normalization for better regularization.
* Experiment with other machine learning models (e.g., Random Forest, XGBoost) for comparison.


## Contributing

Contributions are welcome! If you have suggestions or find issues, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
