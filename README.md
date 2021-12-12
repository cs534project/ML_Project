# ML_Project

## Place the files as: ## 
  
-detection  
&emsp;-Dog_1  
&emsp;&emsp;-xxx.mat  
&emsp;&emsp;...  
&emsp;...  
-prediction  
&emsp;-Dog_1  
&emsp;&emsp;-xxx.mat  
&emsp;&emsp;...  
&emsp;...  
-ML_Project (main project files)  
&emsp;-xx.py  
&emsp;...

--- 

## Some properties of data ##  

### detection ###  

Keys of loaded files:  
<p> &emsp; ['__header__', '__version__', '__globals__', 'data', 'freq', 'channels', 'latency'] </p>
<p> &emsp;&emsp;-- for ictal data </p>
<p> &emsp; ['__header__', '__version__', '__globals__', 'data', 'freq', 'channels'] </p>
<p> &emsp;&emsp;-- for interictal data </p>
<p> Possible to use len(keys()) to determing ictal or interictal</p>  

### prediction ###  
Keys of loaded files:  
<p>&emsp; k = ['__header__', '__version__', '__globals__', 'interictal_segment_1'] </p>
<p>&emsp; k[3] is the real useful part.</p>
<p>&emsp;mat[k[3]][0][0][0] -> Data</p>
<p>&emsp;mat[k[3]][0][0][1] -> len(Data)</p>
<p>&emsp;mat[k[3]][0][0][2] -> Sampleing freq.</p>
<p>&emsp;mat[k[3]][0][0][3] -> channels</p>
<p>&emsp;mat[k[3]][0][0][4] -> sequency = s, then it from segment (s-1)x10 - sx10</p>

## Feature Extraction ##
- Mean, Max, Min, Stdv  
&emsp; Normalize with $$ X^{'} = \frac{X-\mu}{\sigma}$$  
- FFT   
&emsp;Facing Problem!  