# Fairness Metrics

This document is a fairness-metrics cheatsheet, it doesn't go deeply in explanations. 

## Statistical Parity - Disparate Impact - Demographic Parity

The rates of value-1 predictions from the different groups must be equal.
Independence between the predictor and the protected variable.
- S binary <br>
$P(f(X)=1|S=0) = P(f(X)=1|S=1)$

- S continuous or discrete <br>
$P(f(X)=1|S) = P(f(X)=1)$

## Avoiding of Disparate Treatment

The probability that an input leads to prediction 1 should be equal regardless of the value of the sensitive variable.
- S binary <br>
$P(f(X)=1|X_S=x,S=0) = P(f(X)=1|X_S=x,S=1)$ <br>
where $X_S$ represents $X$ without the sensitive variable.


## Equality of Odds

The rates of true and false predictions from the different groups must be equal.
Independence between the error of the model and the protected variable.
- S binary <br>
$P(f(X)=1|Y=i,S=0) = P(f(X)=1|Y=i,S=1) ,i=0,1$

- S general <br>
$P(f(X)=1|Y=i,S) = P(f(X)=1|Y=i) ,i=0,1$

## Avoiding of Disparate Mistreatment

The probability that a prediction is false should be equal regardless of the value of the sensitive variable.
- S binary <br>
$P(f(X)\ne Y|S=1) = P(f(X)\ne Y|S=0)$ <br>


## Global Sensitivity Analysis

GSA is used for quantifying the influence of a set of features on the outcome.<br>
Sobol' indices are based on correlations and need access to the function while CVM' indices are based on rank and need only a sample of evaluations.

**Sobol' indices**<br>
4 indices that quantify how much of the output variance can be explained by the variance of Xi.

|           | Correlation Between Variables  | Joined Contributions | 
|-----------| :----------------------------: |:--------------------:| 
|**$Sob_i$** | ✔️ | ❌ | 
|**$SobT_i$** | ✔️ | ✔️ | 
|**$Sob_i^{ind}$** | ❌ | ❌ |
|**$SobT_i^{ind}$**| ❌ | ✔️ | 

**Cramer-Von Mises' indices**<br>
The 2 CVM' indices is an extension of the Sobol’ indices to quantify more than just the second-order influence of the inputs on the output.

[For further details about GSA in Fairness](https://hal.archives-ouvertes.fr/hal-03160697/file/Fairness_seen_as_GSA.pdf "Fairness seen as Global Sensitivity Analysis")



## Case-of-use Recap

|           | Disparate Impact  | Avoiding Disparate Treatment | Equality Odds | Avoiding Disparate Mistreatment | Sobol' indices | Cramer-Von Mises' indices |
|-----------| :---------------: |:----------------------------:| :------------:| :------------------------------:| :-------------:| :------------------------:|
|**S binary** | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
|**S discrete**| ✔️ | ❌ | ✔️ | ❌ | ✔️ | ✔️ |
|**S continuous**| ✔️ | ❌ | ✔️ | ❌ | ✔️ | ✔️ |
