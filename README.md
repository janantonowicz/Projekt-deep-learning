# Fashion-Mnist-two-models-with-comparison
Project includes two neural networks one very simple and one with data augmentation, trained on fashion mnist dataset. The website build in flask can prompt two models with random image from dataset and display comparison between two models.

Simple model was build using only 5 epochs because going over results in overfitting:</br>
5 epochs:
<img width="945" height="357" alt="5 epochs img" src="https://github.com/user-attachments/assets/d672a9fc-34ce-4cde-a178-a9c2eb218d69" />
10 epochs:
<img width="990" height="374" alt="10 epochs img" src="https://github.com/user-attachments/assets/a87ca1c9-3047-4148-8aef-3a4953c1f44d" />

With augmentation I think the optimal number of epochs is 50:</br>
40 epochs:
<img width="945" height="357" alt="obraz" src="https://github.com/user-attachments/assets/958374bf-2477-4fa8-bb85-919806181c78" />

50 epochs:
<img width="945" height="357" alt="obraz" src="https://github.com/user-attachments/assets/3bcc74cf-d60c-455a-b0ba-5d6d2b38d764" />
