# 1. Penguin-Project

2. By: Ryan Voda, Christen Tai, Raashi Chaudhari

3. We determined a small set of measurements that are highly predictive of a penguin’s species and made a Python program that uses machine learning (random forest classifier and logistic regression) to predict the species of penguin given three predictor variables from penguin data.

4. Instructions on how to install the package requirements: 

   conda create --name NEWENV --file requirements.txt
   
5. The demo file, PenguinProjectWalkthrough.ipynb, provides a step-by-step walkthrough of what code to run and discussion of what each block of code does.

   It begins with data exploration, where we isolate columns and create visualizations which explain
   why we end up making predictions based on species, culmen length, and culmen depth. For example,

   ![image](https://user-images.githubusercontent.com/97138009/158109283-01360e86-38d4-4e03-ba24-0bfab4d79b59.png)

   shows us that the qualitative feature of a penguin's island is useful in deducing a penguin's species.
   
   After exploring the data, we define class penguinData() which Reads in csv file of penguin data 
   and cleans up for Species Prediction by functions logisticRegression() or randomForestClassifier().
   
   Then, we create two instances of exception handling, the first trying to read our csv file, and the second ensuring that the csv is useable.
   
   Now we define function logisticRegression(), and when ran with our valid csv as shown, the following output is expected:
   
   ![image](https://user-images.githubusercontent.com/97138009/158109892-e5298af7-2649-4e92-afe2-f0d0b0743332.png)
   
   Now we define function randomForestClassifier(), and when ran with our valid csv as shown, the following output is expected:
   
   ![image](https://user-images.githubusercontent.com/97138009/158109953-c60351ee-192a-4f61-8296-3a75de130177.png)
   
   We now create our final clustering model, which should look like:
   
   ![image](https://user-images.githubusercontent.com/97138009/158110031-75819598-0f84-4e62-acfe-a2d061884a33.png)

   We conclude with a discussion of our models.
   
   Like we said earlier, PenguinProjectWalkthrough.ipynb provides a very in-depth walkthrough of our code, this is just a summary of it.
   
6. Scope and Limitations:

   This project should not be used to predict penguin species for malicious purposes, and we do not condone any harm towards penguins. Furthermore, this code should not be
   adapted for malicious purposes which extend past its intended use for penguins.
   
   This code is limited and intended to only predict penguin species based on a dataset contained in a csv.
   
   Potential extensions could include:
   
   Adjusting the class to accept and read data in non-csv formats
   
   Creating more functions to produce more models
   
   Adapting everything to make predictions of any animal species based on given data.
   
   Etc.
   
7. License and Terms of Use

   Copyright (c) 2022 Christen Tai, Ryan Voda, Raashi Chaudhari

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
   
   The Palmer Penguins data set was collected by collected by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.
   
8. References and Acknowledgement

   Credit to Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network, for the Palmer Penguins data set.
   
   Credit to Harlin Lee, Vishnu Bachupally.
   
9. The Palmer Penguins data set was collected by collected by Dr. Kristen Gorman and the Palmer Station, Antarctica LTER, a member of the Long Term Ecological Research Network.

10. https://stackoverflow.com/questions/2984888/check-if-file-has-a-csv-format-with-python

   The above link was used to implement Exception handling for ensuring that our given csv file was appropriate. Our code adjusts the code given in the suggestions.
