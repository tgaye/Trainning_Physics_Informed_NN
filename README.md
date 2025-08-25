
====================================================================================================
HYPERPARAMETER STUDY SUMMARY
====================================================================================================
Rank Configuration        Architecture Activation LR       Rel L2 Error Time(s)  Params  
----------------------------------------------------------------------------------------------------
1    Small_Tanh           2x32x32x1    TANH       1e-03    0.4088       128.9    1,185
2    Medium_Tanh_LowLR    2x64x64x1    TANH       1e-04    0.4098       127.1    4,417
3    Medium_Tanh          2x64x64x1    TANH       1e-03    0.4100       124.5    4,417
4    Medium_Tanh_HighLR   2x64x64x1    TANH       5e-03    0.4102       128.0    4,417
5    Medium_GELU          2x64x64x1    GELU       1e-03    0.4103       149.0    4,417
6    Medium_Swish         2x64x64x1    SWISH      1e-03    0.4103       164.7    4,417
7    Large_Tanh           2x128x128x1  TANH       1e-03    0.4112       129.6    17,025
8    Deep_Tanh            2x64x64x64x1 TANH       1e-03    0.4132       159.2    8,577   

🏆 BEST PERFORMER: Small_Tanh
   Architecture: 2x32x32x1
   Activation: TANH
   Relative L2 Error: 0.4088
   Training Time: 128.9s
   Parameters: 1,185

