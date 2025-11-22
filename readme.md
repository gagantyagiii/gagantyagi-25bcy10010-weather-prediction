
Weather Prediction Mini Project

Predicting Rainfall from Temperature (Lucknow Dataset)

This is a small project I made to understand how machine learning works in real life. I used a weather dataset of Lucknow from 1990 to 2022 and tried to predict how much it might rain based on the average temperature. The idea sounded interesting to me, so I thought of giving it a try.

About the Project

We see weather changing every day, and sometimes we want to know if it may rain or not. Instead of guessing, I decided to train a model using real data. The relationship between temperature and rainfall is not always simple, but even then the model can still make some useful predictions.

The program works like this:

It loads the dataset

Fills any missing values so the model doesn’t get confused

Splits the data into training and testing parts

Trains a linear regression model

Finally, the user can enter a temperature and get the predicted rainfall


I also made a simple webpage using Streamlit so anyone can use it easily.

Tools & Libraries Used

Python

Pandas and NumPy (for working with data)

Scikit-Learn (for the model)

Streamlit (for the web app)

A CSV weather file (Lucknow data)


How to Run

If someone wants to try this project:

pip install pandas numpy scikit-learn streamlit

Then run the app with:

streamlit run app.py

(Here, the file name should be same as the code file.)

What I Learned

This was my first experience trying a prediction model on real weather data. I understood how important it is to clean data properly before using it. I also learned how a machine learning model actually makes predictions instead of just plotting data.

Future Scope

There is still a lot to improve. I want to add more features like humidity, wind, or pressure because rainfall does not depend only on temperature. Maybe even try a different model later and compare results.

Conclusion

It’s a small project, but I feel it’s useful for learning the basics of machine learning and how to make simple web applications. I hope to keep improving it as I learn more.



