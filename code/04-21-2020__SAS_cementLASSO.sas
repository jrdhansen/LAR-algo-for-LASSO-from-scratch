/**********************************************
***********************************************
STAT 6100, Final Project, spring 2020        
Hald cement analysis           
Jared Hansen                              
***********************************************
***********************************************/

options nodate pageno=1; run;

/**********************************************
READING IN THE HALD CEMENT DATA  
***********************************************/
data cement;
	input aluminate silicate ferrite dicalcium heat;
	label aluminate = "Percentage (by weight) of tricalcium aluminate";
	label silicate = "Percentage (by weight) of tricalcium silicate";
	label ferrite = "Percentage (by weight) of tetracalcium alumino";
	label dicalcium = "Percentage (by weight) of dicalcium silicate";
	label heat = "Heat evolve during hardening in calories per gram (RESPONSE)";
	datalines;
 7  26   6  60   78.5
 1  29  15  52   74.3
11  56   8  20  104.3
11  31   8  47   87.6
 7  52   6  33   95.9
11  55   9  22  109.2
 3  71  17   6  102.7
 1  31  22  44   72.5
 2  54  18  22   93.1
21  47   4  26  115.9
 1  40  23  34   83.8
11  66   9  12  113.3
10  68   8  12  109.4
;
run;

*Apply LASSO regression to the data using LOOCV for selection;
title1 "LASSO regression using PRESS: Hald cement data";
proc glmselect data=cement plots=coefficients;
	model heat = aluminate silicate ferrite dicalcium /
	selection=LASSO(choose=PRESS steps=50 LSCOEFFS);
run;



