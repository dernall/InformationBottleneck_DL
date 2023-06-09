# Machine Learning for Reliable Comminications under Coarse quantization

This repo contains files and code for DL project.

The information bottleneck method was introduced by N. Tishby 1999. Since then, there a lot of variation and implementations of this algorithm have appeared. In communication theory, this method could be used to coarsely quantize channel and belief propagation messages in LDPC decoding process. However, problem of memory consumption and computational cost arise, as for large codes, one struggle from curse of dimensionality. Therefore modern machine learning techniques could be used to replace classical IB algorithms and proceeds decoding faster.

Consider Relevant Random Variable $X$ nodulated at the receiver, observed Random Variable $Y$ at the receiver 
and Comressed Random Varaibale $T$:

![My Image](InfBottleneckProblem.png)
![My Image](InfBottleneckProblemCnannel.png)

To compress R.V. $Y$ into $T$ we could use one this methods: 
1) Agglomerative Information Bottleneck
2) Sequential Information Bottleneck
3) KL − Means Algorithm 

For this algorythms we show ability to preserve mutual information $I(X, T)$ and $I(X;Y)$:
![My Image](MutualInfo_ToSNR.png)

As we can observe Agglomerative methods shows perfomance worse than other methods. In following steps we will use symmetric version of Sequential Algorithm

Next we use this Information Bottleneck method to quantize message in Belief Propagation decoder and Run simulations for two Regular LDPC codes with identifiers 
8000.4000.3.483 and 816.55.13 from this cite: http://www.inference.org.uk/mackay/codes/data.html
Results of simulations:
![My Image](BER_N8000.png)
![My Image](BER_N816.png)

Next step we train Neural Network to approximate Lookup tables in order to reduce memory consumtion and procceds messages in one step:
![My Image](Loss_NN.png)

## Responsibilities

1) Mikhail Shvetsov -- Project coordinatior, IB methods researcher 
2) Oleg Nesterenkov -- IB methods researcher, NN implementation
3) Roman Khalikov -- IB methods researcher, Hardware provider
4) Grigoriy Vyaznikov -- Different loss incluence researcher, NN implementation

## Future research

1) Try different approaches to calculate MI (Kozachenko - Leonenko, MINE, etc.)
2) Simulation of BP decoder + NN output
