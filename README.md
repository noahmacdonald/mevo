# mevo
Web-app interface for a predictive model of the US auto market.

<img src="https://github.com/noahmacdonald/mevo/blob/master/mevo1.png">

After developing a model implementing the Random Forest algorithm (scikit-learn), I built this simple web app interface for it. Using a Flask backend, the core components of the model implementation in Python are run and results are displayed on the screen. The data is displayed on the left side of the screen (though, no data has been stored in this repo) and each cell is edittable by double clicking. The changes are reflected in the model when the "Run Model" button is pressed. 

Re-Calibrate - Erase extrapolated data.
Best Case - Extrapolate data for the best-case.
Worst Case - Extrapolate data for the worst-case.

Our model's predictions are based on extrapolated data, which is why this project's predictions won't scale beyond ~6 months. Because of this however, predictions are able to reflect different possible future sceanarios. Our "best case" sceanario was based on optimistic forecasts of the model's variables. This best case scenario's predictions were ultimately 95% accurate in predicting 2019Q2 units of auto sales. The "worst case" model was based on dramatizing several indicators of a recession that were already present in the data. Ultimately, we determined the Best Case model to be more likely. 

The dataset we used for this model was a set of predictive indicators our team determined from the ME Survey data at large. Predictive indicators were determined through a tuning process utilizing the Random Forest algorithm, as well as dimensionality reduction techniques like PCA. We also used several variables we determined to be economic indicators such as Fed Funds Rate and GDP. These variables were collected for each month from November 2015 to December 2019 and the amount of cars sold (according to marklines) was the response variable. While training our model, we used the 80% data from 11/2015 to 3/2019 (for that was as far as the survey data went at the time) to train, and 20% of the data from that time period to test. When using the model beyond March, 2019, perhaps as to predict Q2 auto sales - the model uses the extrapolated data from those months to form a prediction. The images displayed on the web application may help to distinguish these two models. 

// Usage

run command:
$ python3 application.py

in browser, go to: 127.0.0.1:5000
