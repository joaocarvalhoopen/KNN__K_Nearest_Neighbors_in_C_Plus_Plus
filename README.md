# KNN K-Nearest-Neighbors in C++
Machine Learning in the edge for IoT (ESP32 or others)<br>

## Description
This is a implementation of the KNN K-Nearest-Neighbors machine learning algorithm in C++. The primary goal is to have a simple, tested and fast implementation of KNN that can be used in IoT (Internet of Things) devices, in my case the target is the ESP32 from Espressif. <br>
ESP32 is a low cost microcontroller that has 2 cores at 240 MHz, 500K RAM, WIFI and Bluetooth. Can be programmed with the Espressif SDK-IDF, the Arduino IDE, MicroPython and others. <br> 
In this implementation of the simple but powerful algorithm KNN, we will pay attention to the space used by the implementation in RAM and to the cache access patterns in it's execution. <br>
The code will be developed and trained here and then in principle used already tested in the microcontroller. We will train and test the KNN code on the public Iris flower dataset. The input data X_Train will be floats and the Y will a categorical int data type. In case you need to have categorical input data dimensions, please do one hot encoding with a float 0.0 and 1.0. It can also be Normalized. <br>
The Iris Dataset comes from [https://archive.ics.uci.edu/ml/datasets/iris](https://archive.ics.uci.edu/ml/datasets/iris) <br> 
The code [KNN__K_Nearest_Neighbors.cpp](./KNN__K_Nearest_Neighbors.cpp) <br>

## Image
![KNN__K_Nearest_Neighbors_img](./KNN__K_Nearest_Neighbors_img.png?raw=true "KNN__K_Nearest_Neighbors_img") <br>

## References
* [How kNN algorithm works - Thales Sehn Korting](https://www.youtube.com/watch?v=UqYde-LULfs) <br>

## License
MIT Open Source License

## Have fun!
Best regards, <br>
Joao Nuno Carvalho <br>