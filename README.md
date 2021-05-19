## Regression Analysis Modeling Based-On Optimization (R.A.M.B.O) ##

Covers topics in regression analysis, such as linear regression, logistic regression, B-spline, and support vector machine. In general, each regression method will cover two optimization methods: one through machine learning optimzaition method, and one through statistical method

## Data Load, Display, and Migrate ##
- Data load features:
  - [X] Load data in 2D format (one independent and one dependent variable)
  - [X] Load data in 3D format (three vector arrays)
- Data display features:
  - [X] Display data as a 2D plot
  - [X] Display data as a 2D plot with colors
  - [X] Display data as a 3D surface plot

## Himmelblau's Function: ##
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;f(x,y)&space;=&space;(x^2&space;&plus;&space;y&space;-&space;11)^2&space;&plus;&space;(x&space;&plus;&space;y^2&space;-&space;7)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;f(x,y)&space;=&space;(x^2&space;&plus;&space;y&space;-&space;11)^2&space;&plus;&space;(x&space;&plus;&space;y^2&space;-&space;7)^2" title="\large f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2" /></a>
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\frac{df(x,y)}{dx}&space;=&space;4x^3&space;&plus;&space;4x(y-11)&space;&plus;&space;2x&plus;2(y^2-7)&space;\\&space;\frac{df(x,y)}{dy}&space;=&space;2x^2&space;&plus;&space;4y^3&space;&plus;4xy&space;-26y&space;-&space;22" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\large&space;\frac{df(x,y)}{dx}&space;=&space;4x^3&space;&plus;&space;4x(y-11)&space;&plus;&space;2x&plus;2(y^2-7)&space;\\&space;\frac{df(x,y)}{dy}&space;=&space;2x^2&space;&plus;&space;4y^3&space;&plus;4xy&space;-26y&space;-&space;22" title="\large \frac{df(x,y)}{dx} = 4x^3 + 4x(y-11) + 2x+2(y^2-7) \\ \frac{df(x,y)}{dy} = 2x^2 + 4y^3 +4xy -26y - 22" /></a>
<br />

<table> <tr>
<th> Two-Dimensional Plot </th> <th> Three-Dimensional Plot </th>
</tr>
<tr>
<td> <img src='./assets/himmelblau_2D.png'> </td>
<td> <img src='./assets/himmelblau_3D.png'> </td>
</tr> </table>

## Linear Regression ##
- [X] Using statistics approach (R-square value)
- [X] Using machine learning approach (RMSE value)
- [X] Using TensorFlow and PyTorch

<table> <tr>
<th> Statistical Analysis </th> <th> Machine Learning </th>
</tr>
<tr>
<td> <img src='./assets/rsquare.png'> </td>
<td> <img src='./assets/rmse.png'> </td>
<tr>
<td> m = 0.498; b = 0.725 </td>
<td> m = 0.559; b = 0.022 </td>
</tr> </table>

## Stanford Code in Place Spring 2021 - COVID19 Analysis ##
<th> Statistical Analysis </th> <th> Machine Learning </th>
</tr>
<tr>
<td> <img src='./assets/covid_statistics.png' width=40% height=40%> </td>
<td> <img src='./assets/covid_machine_learning.png' width=40% height=40%> </td>
<tr>
<td> m = 89.8580; b = 66040.458 </td>
<td> m = 112.245; b = 47667.073 </td>
</tr> </table>


## Support Vector Machine ##
- [ ] Binary classification purpose
- [X] Determine the hyper-plane
<table> <tr>
<th> Initial Iteration </th>
<th> Final Iteration </th>
</tr>
<tr>
<td> <img src='./assets/perceptrons/Iteration Number 0.png'> </td>
<td> <img src='./assets/perceptrons/Iteration Number 2190.png'> </td>
</tr> </table>

<br />
<b> Support Vector Machine Hyper Plane Determination </b>
<br />
<p align='center'>
<img src = './assets/svm_hyperplane.gif' width ='500'>
</p>

