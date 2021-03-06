# ODE Phase Planes
An example on how to generate phase plots of the examples given in the pdf.


![Test Image 8](https://github.com/isakhammer/ODE_phase_planes/blob/master/results/problem5.jpg)



```
Let us define the system, X' = F(X) 
⎡x⎤
⎢ ⎥
⎣y⎦
= 
⎡     ⎛   2    2    ⎞  ⎤
⎢  -x⋅⎝- x  - y  + 2⎠  ⎥
⎢                      ⎥
⎢   ⎛ 2          2    ⎞⎥
⎣-y⋅⎝x  - 3⋅x + y  + 1⎠⎦
With the equilibrium points
[{x: 0, y: 0}, {x: 0, y: -ⅈ}, {x: 0, y: ⅈ}, {x: 1, y: -1}, {x: 1, y: 1}, {x: -
√2, y: 0}, {x: √2, y: 0}]

 The F Jacobian is 

⎡   2    2                           ⎤
⎢3⋅x  + y  - 2          2⋅x⋅y        ⎥
⎢                                    ⎥
⎢                  2            2    ⎥
⎣ y⋅(3 - 2⋅x)   - x  + 3⋅x - 3⋅y  - 1⎦

 The classified equilibrium points are 
Eq point  0 is -> 0 0 with DF matrix: 
⎡-2  0 ⎤
⎢      ⎥
⎣0   -1⎦
Point is sink
Eq point  1 is -> 0 -I with DF matrix: 
⎡ -3   0⎤
⎢       ⎥
⎣-3⋅ⅈ  2⎦
Point is saddle
Eq point  2 is -> 0 I with DF matrix: 
⎡-3   0⎤
⎢      ⎥
⎣3⋅ⅈ  2⎦
Point is saddle
Eq point  3 is -> 1 -1 with DF matrix: 
⎡2   -2⎤
⎢      ⎥
⎣-1  -2⎦
Point is saddle
Eq point  4 is -> 1 1 with DF matrix: 
⎡2  2 ⎤
⎢     ⎥
⎣1  -2⎦
Point is saddle
Eq point  5 is -> -sqrt(2) 0 with DF matrix: 
⎡4      0    ⎤
⎢            ⎥
⎣0  -3⋅√2 - 3⎦
Point is saddle
Eq point  6 is -> sqrt(2) 0 with DF matrix: 
⎡4      0    ⎤
⎢            ⎥
⎣0  -3 + 3⋅√2⎦
Point is source

```
