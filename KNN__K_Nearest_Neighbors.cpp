/******************************************************************************
 *                                                                            *
 *  KNN  K-Nearest-Neighbors in C++                                           *
 *                                                                            *
 ******************************************************************************
 * Author: Joao Nuno Carvalho                                                 *
 * Date: 2019.09.22                                                           *
 * License: MIT Open Source License                                           *
 * Description: This is a implementation of the KNN K-Nearest-Neighbors       *
 *              machine learning algorithm in C++. The primary goal           *
 *              is to have a simple, tested and fast implementation of KNN    *
 *              that can be used in IoT (Internet of Things) devices, in my   *
 *              case the target is the ESP32 from Espressif. ESP32 is a low   *
 *              cost microcontroller that has 2 cores at 240 MHz, 500K RAM,   *
 *              WIFI and Bluetooth. Can be programmed with the Espressif      *
 *              SDK-IDF, the Arduino IDE, MicroPython and others.             *
 *              In this implementation of the simple but powerful algorithm   *
 *              KNN, we will pay attention to the space used by the           *
 *              implementation in RAM and to the cache access patterns in     *
 *              it's execution.                                               *
 *              The code will be developed and trainned here and then         *
 *              in principle used already tested in the microcontroller.      *
 *              We will train and test the KNN code on the public Iris flower *
 *              dataset. The input data X_Train will be floats and the Y will *
 *              a categorical int data type. In case you need to have         *
 *              categorical input data dimensions, please do one hot encoding *
 *              with a float 0.0 and 1.0. It can also be Normalized.          *
 *              The Iris Dataset comes from                                   *
 *              https://archive.ics.uci.edu/ml/datasets/iris                  *
 *              For more details on the algorithm see references on           *
 *              project page.                                                 *
 ******************************************************************************
*/  

#include <iostream>
#include <fstream>
#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <math.h>       /* sqrt */
#include <utility>      /* std::pair, std::make_pair */
#include <algorithm>    /* std::count */
#include <chrono>


using namespace std;

const vector<string> vec_class_strings {"Iris-setosa", "Iris-versicolor", "Iris-virginica"};
constexpr int k = 5; 

void printVec(vector<int> vec);

void printVec2d(vector<vector<int>> vec_2d, string vec_name);

void printVec2d_inverted(vector<vector<double>> vec_2d, string vec_name);

// Measuring the calculation duration time of the different methods.
void measure_time(int n);

bool read_data_set(string filename, vector<vector<float>> &vec_X_dataset, vector<int> &vec_Y_dataset){    
    float field0, field1, field2, field3;
    char comma;
    string class_name;

    string line;
    ifstream myfile (filename);
    if (myfile.is_open())
    {
        while(myfile 
              >> field0 >> comma 
              >> field1 >> comma 
              >> field2 >> comma
              >> field3 >> comma
              >> class_name )
        {
/*
            cout << field0 << comma 
                 << field1 << comma 
                 << field2 << comma
                 << field3 << comma
                 << class_name << endl;
*/
            vector<float> inner_vec {field0, field1, field2, field3};
            vec_X_dataset.push_back(inner_vec);
            int y = -1;
            if (class_name == vec_class_strings[0])
                y = 0;
            else if (class_name == vec_class_strings[1])
                y = 1;
            else if (class_name == vec_class_strings[2])
                y = 2;
            vec_Y_dataset.push_back(y);
        }
        myfile.close();

    }else{ 
        cout << "Unable to open file";
        return true;
    } 
    return false;
}

void mix_dataset(vector<vector<float>> &vec_X_dataset, vector<int> &vec_Y_dataset){
    size_t len = vec_X_dataset.size();
    vector<float> data_point;
    for(size_t i = 0; i < len; ++i){
        size_t swap_index = rand() % len;  // Random number between 0 and len-1.
        if (i == swap_index)
            continue;
        
        vector<float> data_point = vec_X_dataset[i];
        vec_X_dataset[i] = vec_X_dataset[swap_index];
        vec_X_dataset[swap_index] = data_point;
        
        int Y = vec_Y_dataset[i];
        vec_Y_dataset[i] = vec_Y_dataset[swap_index];
        vec_Y_dataset[swap_index] = Y;            
    }
}

void split_dataset(int perc_train,
                   vector<vector<float>> &vec_X_dataset, vector<int> &vec_Y_dataset, 
                   vector<vector<float>> &vec_X_train,   vector<int> &vec_Y_train, 
                   vector<vector<float>> &vec_X_test,    vector<int> &vec_Y_test){                    
    size_t len = vec_X_dataset.size();
    size_t division = static_cast<size_t>(len * (perc_train * 0.01));
    vector<float> data_point;
    for(size_t i = 0; i < len; ++i){
        if (i < division){
            vec_X_train.push_back(vec_X_dataset[i]);
            vec_Y_train.push_back(vec_Y_dataset[i]);
        }else{
            vec_X_test.push_back(vec_X_dataset[i]);
            vec_Y_test.push_back(vec_Y_dataset[i]);
        }
    }
}

float distance(vector<float> &point_A, vector<float> &point_B){
    float sum = 0.0;
    for(size_t i = 0; i < point_A.size(); ++i){   
        float a = point_A[i]-point_B[i];
        sum += a*a;
    }
    return sqrt(sum);
}

int KNN_classifier(vector<vector<float>> &vec_X_train, vector<int> &vec_Y_train,
                   int k, vector<float> data_point){
    // Returns y_point.

    // Calculate the distance between data_point and all points.
    vector<pair<int, float>> dist_vec {};
    pair<int, float> pair_tmp;
    for(size_t i = 0; i < vec_X_train.size(); ++i){   
        pair_tmp = make_pair(i, distance(vec_X_train[i], data_point));
        dist_vec.push_back(pair_tmp);
    }

    // Sort dist_vec by distance ascending.
    auto sortRuleLambda = [] (pair<int, float> const& s1, pair<int, float> const& s2) -> bool
    {
       return s1.second < s2.second;
    };
    
    sort(dist_vec.begin(), dist_vec.end(), sortRuleLambda);

    // Majority vote on the Y target int class.
    const size_t num_classes = vec_class_strings.size();
    for(size_t i = k; i > 0; --i){
        vector<int> vec_Y_class_target(num_classes);
        for(size_t c = 0; c < num_classes; ++c){
            // c is the value of the class to be compared to the top K elements,
            // to see if we find a majority choosen class. 
            // If we didn't find we will try to find the majority in the best k-1
            // until we only compare to one and in that case it will be the majority.
            for(size_t j=0; j<i; ++j){
                int index = dist_vec[j].first;
                if (vec_Y_train[index] == static_cast<int>(c))
                    vec_Y_class_target[c]++;
            }
        }

        // Find the maximum value fo the counting histogram.
        float max = -1;
        size_t max_index = -1;
        for(size_t r = 0; r < vec_Y_class_target.size(); ++r){
            int elem = vec_Y_class_target[r];
            if (elem >= max){
                max = elem;
                max_index = r;
            }
        }

        bool flag_continue = false;
        // Find if there is only one max value on the counting histogram.
        for(size_t r = 0; r < vec_Y_class_target.size(); ++r){
            int elem = vec_Y_class_target[r];
            if ((elem == max) && (r != max_index)){
                flag_continue = true;
                break; 
            }
        }

        if (flag_continue == true)
            continue;
        else{
            return static_cast<int>(max_index);
        }
    }

    return -1;
}

void evaluate_all_dataset(vector<vector<float>> &vec_X_train, vector<int> &vec_Y_train, int k,
                          vector<vector<float>> &vec_X_dataset, vector<int> &vec_Y_dataset,
                          int &dataset_len, int &correct_dataset_pred, float &correct_dataset_pred_perc){
    dataset_len = vec_X_dataset.size();
    correct_dataset_pred = 0;
    correct_dataset_pred_perc = 0.0;    
    vector<float> data_point;
    for(size_t i = 0; i < vec_X_dataset.size(); ++i){
        vector<float> data_point = vec_X_dataset[i];
        int y_point = KNN_classifier(vec_X_train, vec_Y_train, k, data_point);
        if (y_point == vec_Y_dataset[i])
            correct_dataset_pred++;
        else{
            cout << data_point[0] << "," << data_point[1] << "," << data_point[2] << "," << data_point[3]
                 << "," << vec_class_strings[y_point] << endl; 
        }        
    }
    correct_dataset_pred_perc = (static_cast<float>(correct_dataset_pred) / dataset_len) * 100;
}

int main(){

    cout << endl << "KNN  K-Nearest-Neighbors in C++" << endl << endl;

    // Read dataset from file.
    string filename = ".//iris.data";
    vector<vector<float>> vec_X_dataset {};
    vector<int> vec_Y_dataset {};
    bool error = read_data_set(filename, vec_X_dataset, vec_Y_dataset);
    if (error){
        cout << "Exiting with error while reading dataset file " << filename << endl;
        exit(-1);
    }

    // Randomly mix the dataset.
    // Initialize the seed.
    srand(3); 
    mix_dataset(vec_X_dataset, vec_Y_dataset);

    // Split the dataset in train and test sets.
    // Train set 80% Test set 20%.
    int perc_train = 80;
    vector<vector<float>> vec_X_train {};
    vector<int> vec_Y_train {};
    vector<vector<float>> vec_X_test {};
    vector<int> vec_Y_test {};
    split_dataset(perc_train, vec_X_dataset, vec_Y_dataset, vec_X_train, vec_Y_train, vec_X_test, vec_Y_test);

    // Evaluate the correct train set percentage.
    int train_len = 0;
    int correct_train_pred = 0;
    float correct_train_pred_perc = 0.0;    
    evaluate_all_dataset(vec_X_train, vec_Y_train, k,
                         vec_X_train, vec_Y_train, // This are the one being tested.
                         train_len, correct_train_pred, correct_train_pred_perc);

    cout << "Correct classification in train set \n\ttrain_len: " << train_len 
         << "\n\t correct_train_pred: "      << correct_train_pred
         << "\n\t correct_train_pred_perc: " << correct_train_pred_perc << endl;

    // Evaluate the correct test set percentage.
    int test_len = 0;
    int correct_test_pred = 0;
    float correct_test_pred_perc = 0.0;    
    evaluate_all_dataset(vec_X_train, vec_Y_train, k,
                         vec_X_test, vec_Y_test,  // This are the one being tested.
                         test_len, correct_test_pred, correct_test_pred_perc);

    cout << "Correct classification in test set \n\ttest_len: " << test_len 
         << "\n\t correct_test_pred: "      << correct_test_pred
         << "\n\t correct_test_pred_perc: " << correct_test_pred_perc << endl;

    return 0;
}


void printVec(vector<int> vec){
    cout << "{";
    bool first_flag {true};
    for(auto elem : vec){
        cout << ((first_flag)? "": ", ") << elem;
        first_flag = false;
    }
    cout << "}" << endl;
}

void printVec2d(vector<vector<int>> vec_2d, string vec_name){
    cout << vec_name << endl;
    for(auto inner_vec : vec_2d){
        for(auto elem : inner_vec){
            cout << elem << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void printVec2d_inverted(vector<vector<double>> vec_2d, string vec_name){
    cout << vec_name << endl;
    for(auto r = vec_2d.end()-1; r >= vec_2d.begin(); --r){
        for(auto c = r->begin(); c < r->end(); ++c){
            //cout << static_cast<double>(*c) << " ";
            cout << *c << " ";
        }
        cout << endl;
    }
    cout << endl;
}

// Measuring the calculation duration time of the different methods.
void measure_time(int n){
    cout << "Measure time" << endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    // Function A to test.
    // function_A();
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration_a = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    cout << "time duration - function_A( ) : " << duration_a << " micro seconds" << endl;

    t1 = std::chrono::high_resolution_clock::now();
    // Function B to test.
    // function_B();
    t2 = std::chrono::high_resolution_clock::now();
    duration_a = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
    cout << "time duration - function_B( ) : " << duration_a << " micro seconds" << endl;
}
