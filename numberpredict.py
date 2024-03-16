#Code done on Google Colab
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
data=mnist.load_data()

(X_train,Y_train),(X_test,Y_test)=data
X_train[0].shape

plt.imshow(X_test[0],cmap='gray')
plt.title('First Image in Test set')
plt.axis('off')
plt.show()

X_train=X_train.reshape((X_train.shape[0],28*28)).astype('float32')
X_test=X_test.reshape((X_test.shape[0],28*28)).astype('float32')

X_train=X_train/255
X_test=X_test/255

from keras.utils import to_categorical

Y_train=to_categorical(Y_train)
Y_test=to_categorical(Y_test)

from keras.models import Sequential
from keras.layers import Dense
model=Sequential()
model.add(Dense(32,input_dim=28*28,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])

model.summary()

model.fit(X_train,Y_train,epochs=30,batch_size=100,validation_data=(X_test,Y_test))

import tensorflow.keras as keras
from PIL import Image
import numpy as np

image="/content/testimgtwo.png"
image=Image.open(image).convert('L')
image=image.resize((28,28))
image

plt.imshow(image,cmap='gray')
plt.axis('off')
plt.show()

image=np.array(image)
image=image.reshape(1,28*28)
image=image.astype('float32')/255

predictions=model.predict(image)
predicted_class=np.argmax(predictions)
print("Predicted class: ",predicted_class)