import matplotlib.pyplot as plt
import pandas as pd

def plot_history(history):
    if  'DataFrame' in str(type(history)):
        loss = history['loss']
        val_loss = history['val_loss']
        
        # 정확도 추출
        accuracy = history['accuracy']
        val_accuracy = history['val_accuracy']
    else:
        # 손실 값 추출
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # 정확도 추출
        accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

    # 그래프 그리기
    epochs = range(1, len(loss) + 1)

    plt.figure(figsize=(12, 5))

    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, 'b', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


    
if __name__ == 'main':
    path = './save_model/03-25-2024_01-49-24/model_history.csv'
    # history 객체를 이용하여 그래프 그리기
    history = pd.read_csv(path)
    plot_history(history)