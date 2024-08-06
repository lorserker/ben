# Single Dummy estimator

The single dummy model is an estimator, that estimates number of tricks for declarer given all 4 hands.

It is trained using double dummy analysis, but not as precise as a real double dummy solver.

There is two different models. One trained by Lorand based on Jack data, and needs an opening lead as input. And another trained by Thorvald, that is based on 10 mill deals, where the double dummy is calculated by Richard Pavlicek.

In the configuration the value of lead_included defines the net to be used:

[eval]  
single_dummy_estimator = UCBC 2024/Models/lr3-1000000  
double_dummy_estimator = UCBC 2024/Models/single_dummy-32768000  
estimator = dde
