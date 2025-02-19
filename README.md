# Hazard-Detection-for-Autonomous-Vehicles


---

## Overview  
This project focuses on **trajectory prediction and hazard detection** using deep learning and ensemble machine learning models. It leverages data from the **KITTI-360 Dataset** ([Dataset Link](https://www.cvlibs.net/datasets/kitti-360/index.php)) and applies multiple predictive models, including:  

- **YOLOv5** for object detection  
- **LSTM (Long Short-Term Memory)** for trajectory prediction  
- **Random Forest, XGBoost, and SVM** for hazard prediction  
- **Voting Ensemble Model** for combining predictions from multiple models  

## Dependencies  
Ensure you have the following libraries installed:  
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow torch opencv-python xgboost ultralytics
```

## Features  
- **Data Loading**: Loads KITTI/Waymo dataset annotations from a CSV file.  
- **Object Detection**: Uses YOLOv5 for detecting vehicles, pedestrians, and obstacles.  
- **Trajectory Prediction**: Implements an LSTM model to predict future positions of moving objects.  
- **Hazard Detection**: Trains multiple regression models (Random Forest, XGBoost, SVM) to evaluate risks.  
- **Ensemble Learning**: Combines predictions using a **Voting Regressor** for improved accuracy.  
- **Performance Evaluation**: Computes **Mean Squared Error (MSE)** and **R² Score** for all models.  
- **Visualization**: Plots **Actual vs Predicted Trajectories** for different models.  

## Dataset  
The project uses the **KITTI-360 Dataset**, which contains images and labeled data for autonomous driving applications. You need to download and extract the dataset before running the script.  

## Model Implementations  

### 1. **Object Detection (YOLOv5)**  
```python
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
```
Detects objects in images using a pre-trained YOLOv5 model.  

### 2. **Trajectory Prediction (LSTM Model)**  
```python
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
```
Predicts future trajectories using LSTM.  

### 3. **Hazard Prediction Models**  
- **Random Forest**  
- **XGBoost**  
- **Support Vector Machine (SVM)**  
- **Ensemble Model (Voting Regressor)**  

```python
ensemble_model = VotingRegressor(
    estimators=[('lstm', lstm_model), ('rf', rf_model), ('xgb', xgb_model), ('svm', svm_model)]
)
ensemble_model.fit(X_train, y_train)
```

## Performance Evaluation  
The models are evaluated using:  
- **Mean Squared Error (MSE)**  
- **R² Score (Coefficient of Determination)**  

Example Output:  
```
LSTM MSE: 0.025, R²: 0.89  
Random Forest MSE: 0.018, R²: 0.92  
XGBoost MSE: 0.020, R²: 0.91  
SVM MSE: 0.030, R²: 0.85  
Ensemble Model MSE: 0.015, R²: 0.94  
```

## Visualization  
The project generates scatter plots comparing **actual vs predicted** trajectories for different models.  

```python
plt.scatter(y_test, ensemble_preds, color='orange')
plt.title('Ensemble Model Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
```

## Future Improvements  
- Integrating **3D LiDAR data** for enhanced perception.  
- Using **Transformer-based models** for trajectory prediction.  
- Implementing **real-time inference** on embedded systems (Jetson, Raspberry Pi).  

## References  
- **KITTI-360 Dataset**: [KITTI Official Site](https://www.cvlibs.net/datasets/kitti-360/index.php)  
- **YOLOv5 Documentation**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)  
- **TensorFlow LSTM**: [TensorFlow Guide](https://www.tensorflow.org/guide)  

---

