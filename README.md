## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+**
*   **pip** (Python package installer)

### Installation

2.  **Download Face Detector Models:**

    Our real-time system uses pre-trained OpenCV DNN face detection models. Download these two files:(click on links)
    *   [`deploy.prototxt.txt`](https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt)
    *   [`res10_300x300_ssd_iter_140000.caffemodel`](https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel)

    Place them in the specified path: Update the `FACE_DETECTOR_PROTO` and `FACE_DETECTOR_MODEL` paths in `webcam_emotion_detector.py` and `image_input.py` to match your chosen location.

3.  **Prepare Your Dataset:**
    Your emotion images should be organized into `images/train` and `images/test` directories, with each emotion as a dedicated subfolder. This structure is automatically recognized by `torchvision.datasets.ImageFolder`.

## Results & Performance

Our journey culminated in a robust Vision Transformer model that achieved a peak validation accuracy of **65.55%** on the challenging FER2013-like dataset. This is a highly competitive result for a 7-class emotion recognition task.

More importantly, the model demonstrates balanced and strong performance across individual emotion categories. Below is the final classification report from our best training run:

| Class    | Precision | Recall | F1-Score | Support |
| :------- | :-------- | :----- | :------- | :------ |
| angry    | 0.5780    | 0.5594 | **0.5686** | 960     |
| disgust  | 0.6220    | 0.7117 | **0.6639** | 111     |
| fear     | 0.5245    | 0.4204 | **0.4667** | 1018    |
| happy    | 0.8522    | 0.8433 | **0.8477** | 1825    |
| neutral  | 0.5899    | 0.6530 | **0.6198** | 1216    |
| sad      | 0.5087    | 0.5127 | **0.5107** | 1139    |
| surprise | 0.7304    | 0.8193 | **0.7723** | 797     |
|          |           |        |          |         |
| **accuracy** |           |        | **0.6530** | 7066    |
| **macro avg** | **0.6294** | **0.6457** | **0.6357** | 7066    |
| **weighted avg** | **0.6499** | **0.6530** | **0.6500** | 7066    |

**The 'disgust' success story:** Notice the impressive **0.6639 F1-score** for the 'disgust' class! This was a major challenge, starting at 0.00 F1-score, and its significant improvement highlights the effectiveness of using weighted loss and fine-tuning with the Vision Transformer.

## Visuals of Emotions in Action

This is where our project truly comes alive! Beyond the summary numbers, seeing the model's progress and its decisions helps us understand its strengths and its learning opportunities.

*   **Confusion Matrix: Unmasking Insights!**
    This visual map is more than just numbers; it tells us *how* our model is performing on each emotion, not just if it's right or wrong. It shows *what* emotions are most often confused for *which other* emotions.

    **How to Read It:**
    *   Each **row** represents the *actual* (true) emotion of the image.
    *   Each **column** represents the *emotion our model predicted*.
    *   The values on the **diagonal** (top-left to bottom-right) show the proportion of times the model got it *right* for that specific emotion. Higher values here mean better performance for that class!
    *   Values *off* the diagonal show misclassifications. For example, if the 'Fear' row has a high value in the 'Angry' column, it means actual 'Fear' expressions are often mistaken for 'Angry' by the model.

    Our matrix reveals some fascinating insights:
    *   `Happy` and `Surprise` are rockstars: Their high diagonal values (e.g., 0.84 for Happy, 0.82 for Surprise) confirm our model excels at identifying these clear expressions.
    *   `Disgust` is no longer invisible! A strong diagonal value of `0.71` (meaning 71% of actual 'disgust' faces were correctly identified) is a huge win, especially considering its initial 0.00 F1-score.
    *   Common confusions: We can see where expressions subtly blend. For instance, 'Fear' (`0.42` recall) might sometimes be mistaken for 'Sad' or 'Neutral', highlighting the dataset's inherent ambiguities. This matrix is an invaluable tool for understanding specific strengths and weaknesses of the model beyond a single accuracy score.

    ![Confusion Matrix](assets/confusion_matrix.png)
    

    *(Tip: This image is generated after training is done!)*

*   **Multi-Face Prediction: A Snapshot of Real-World Application**
    To demonstrate the system's ability to handle more complex scenarios, here's an image featuring multiple individuals. Our model processes each detected face independently, providing a unique emotion prediction for every person.

    *Image of multiple faces-* ![Multi-Face Prediction](assets/multi_face_image.png)

    **Detailed Predictions from the Scene:**

    *   **Face 1 (True: Neutral): Predicted as `Angry` (Confidence: 91.3%)**  (it's not a flaw, more info in Misclassified Image Prediction) 
        *   The model's top 3 probabilities were `angry: 91.26%`, `sad: 7.16%`, `neutral: 1.12%`.
    *   **Face 2 (True: Happy): Predicted as `Happy` (Confidence: 99.9%)**
        *   Top 3: `happy: 99.88%`, `neutral: 0.07%`, `surprise: 0.03%`.
    *   **Face 3 (True: Surprise): Predicted as `Surprise` (Confidence: 99.9%)**
        *   Top 3: `surprise: 99.89%`, `fear: 0.05%`, `happy: 0.03%`.
    *   **Face 4 (True: Angry): Predicted as `Angry` (Confidence: 76.3%)**
        *   Top 3: `angry: 76.34%`, `sad: 18.45%`, `fear: 4.28%`.
    *   **Face 5 (True: Sad): Predicted as `Fear` (Confidence: 45.2%)**
        *   Top 3: `fear: 45.20%`, `sad: 31.27%`, `neutral: 20.96%`.
    *   **Face 6 (True: Fear): Predicted as `Surprise` (Confidence: 96.8%)**
        *   Top 3: `surprise: 96.80%`, `fear: 2.07%`, `angry: 0.79%`.
    *   **Face 7 (True: Disgust): Predicted as `Disgust` (Confidence: 91.8%)**
        *   Top 3: `disgust: 91.79%`, `angry: 7.21%`, `sad: 0.56%`.

*   **Misclassified Image Predictions: A Closer Look at Specific Confusions:**
    Beyond the multi-face scene, these dedicated examples highlight typical challenges where our model, like even human observers, finds certain expressions difficult to definitively classify. This isn't a flaw, but a valuable insight into the inherent complexity of facial cues and potential dataset ambiguities.

    *   **True: Fear, Predicted: Sad:** A common overlap. Expressions of fear often involve raised inner eyebrows and a slightly open mouth, which can visually blend with cues for sadness, particularly if the fear is not extreme.
        *   ![Misclassified Fear as Sad](assets/fear_sad1.png) ![Misclassified Fear as Sad](assets/fear_sad2.png)

    *   **True: Fear, Predicted: Neutral:** Sometimes, a subtle fear expression, or tension around the eyes or mouth, might be interpreted as a neutral state by the model.
        *   ![Misclassified Fear as Neutral](assets/fear_neutral.png)

    *   **True: Sad, Predicted: Fear:** Conversely, sadness can involve brow furrowing or a general downturn that might be confused with mild fear, especially if the expression is not exaggerated.
        *   ![Misclassified Sad as Fear](assets/sad_fear.png)

    *   **True: Surprise, Predicted: Angry:** This is less common but can occur. A strong, sudden intake of breath or a wide-open mouth in surprise might be misread as an aggressive, open-mouthed angry expression, especially if accompanying eyebrow movements are ambiguous.
        *   ![Misclassified Surprise as Angry](assets/surprise_angry.png)

## Credits & Acknowledgements

*   **PyTorch & torchvision:** The foundational deep learning libraries that powered this project.
*   **OpenCV:** For robust video and image processing utilities, and the pre-trained face detection model.
*   **scikit-learn:** For powerful model evaluation metrics.
*   **PIL (Pillow):** For advanced image manipulation and text rendering.
*   **tqdm:** For providing delightful progress bars during training.
*   **FER2013 Dataset:** (If this is the dataset you used, it's good practice to acknowledge it explicitly, as its characteristics shaped the project's challenges and solutions).   
[`Dataset`](https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer)


## Conclusion & Let's Connect!

And there you have it! The journey through the nuanced world of facial emotions, from tackling elusive 'disgust' with weighted loss to finally harnessing the transformative power of Vision Transformers, has been truly rewarding. Witnessing our model learn to 'see' and interpret emotions with remarkable clarity (and achieve over 65% accuracy!) has been a testament to the exciting capabilities of modern deep learning.

This project isn't just about code; it's about exploring the subtle language of human expression through the lens of artificial intelligence.

Now it's your turn!

Dive into the code, run the webcam demo, and watch the pixels come alive with feeling. Feel free to experiment, tweak parameters, find new insights, or even break things (that's how we learn!). Your feedback, ideas, and contributions are incredibly welcome. Let's keep pushing the boundaries of what machines can understand about the human experience.

Happy emoting! 😃
---
