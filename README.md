# AGP-152-NonQuant

A web-based application that predicts a person's age and gender from a facial image using a TensorFlow.js model. The application is hosted on Firebase, enabling real-time inference directly in the browser without the need for server-side computation.

**MODEL**: ResNet152 (Non Quantised)
**WEBSITE LINK**: https://agp-152-non-quantised.web.app

---

## 🚀 Features

- **Real-Time Predictions**: Utilizes a pre-trained ResNet model converted to TensorFlow.js for instant age and gender predictions in the browser.
- **Edge Deployment**: Runs entirely on the client-side, ensuring user privacy and eliminating server dependencies.
- **Firebase Hosting**: Seamless deployment and hosting using Firebase, providing a scalable and reliable platform for the application.
- **User-Friendly Interface**: Simple and intuitive UI built with HTML, CSS, and JavaScript.

---

## 📁 Project Structure

```
age-gender-predictor/
├── tfjs_model/           # TensorFlow.js model files
├── tfjs_model_temp/      # Temporary Folder containing unused model
├── index.html            # Main HTML file
├── app.js                # JavaScript logic for handling predictions
├── app.css               # Styling for the application
├── firebase.json         # Firebase configuration
├── .firebaserc           # Firebase project settings
├── .gitignore            # Git ignore file
├── LICENSE               # MIT License
└── README.md             # Project documentation
```

---

## 🛠️ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Irteza-faisal/age-gender-predictor.git
   cd age-gender-predictor
   ```

2. **Install Firebase CLI**:
   Ensure you have the Firebase CLI installed. If not, install it using npm:
   ```bash
   npm install -g firebase-tools
   ```

3. **Login to Firebase**:
   Authenticate your Firebase account:
   ```bash
   firebase login
   ```

4. **Initialize Firebase Project**:
   If you haven't already, initialize your Firebase project:
   ```bash
   firebase init
   ```

5. **Deploy to Firebase**:
   Deploy the application to Firebase Hosting:
   ```bash
   firebase deploy
   ```

---

## 🔍 Usage

1. **Access the Application**:
   Open the deployed Firebase URL in your browser.

2. **Upload an Image**:
   Use the provided interface to upload a facial image.

3. **View Predictions**:
   The application will display the predicted age and gender based on the uploaded image.

---

## 🤖 Model Details

- **Architecture**: ResNet model converted to TensorFlow.js format.
- **Functionality**: Processes facial images to predict age and gender.
- **Deployment**: Optimized for client-side inference using TensorFlow.js.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Inspired by research in age and gender prediction using deep convolutional neural networks and transfer learning.
- Utilizes Firebase for hosting and deployment.

---

For more information, visit the [GitHub Repository](https://github.com/Irteza-faisal/AGP-152-NonQuant).
