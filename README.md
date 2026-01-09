# Fashion-Mnist-two-models-with-comparison
This project features two neural network models trained on the Fashion MNIST dataset: a basic model and a data-augmented model. The web application, developed using Flask, allows users to randomly sample images from the dataset and compare predictions made by both models side by side.
Neural Network Models

    Simple Model:
        This model was implemented with only 5 training epochs to mitigate overfitting issues. Despite its simplicity, it serves as a baseline for performance comparison.

    5 epochs:
    <img width="945" height="357" alt="5 epochs img" src="https://github.com/user-attachments/assets/d672a9fc-34ce-4cde-a178-a9c2eb218d69" />
    10 epochs:
    <img width="990" height="374" alt="10 epochs img" src="https://github.com/user-attachments/assets/a87ca1c9-3047-4148-8aef-3a4953c1f44d" />
    
    Data Augmented Model:
        This neural network incorporates techniques such as rotation, zooming, and flipping to enhance the dataset and improve accuracy and robustness.

      40 epochs:
      <img width="945" height="357" alt="obraz" src="https://github.com/user-attachments/assets/958374bf-2477-4fa8-bb85-919806181c78" />
      
      50 epochs:
      <img width="945" height="357" alt="obraz" src="https://github.com/user-attachments/assets/3bcc74cf-d60c-455a-b0ba-5d6d2b38d764" />

Web Application Features

    Random Image Sampling:
        Users can view randomly selected images from the Fashion MNIST dataset.

    Model Comparison:
        The application displays predictions from both models, allowing for immediate visual comparison of their performances.
