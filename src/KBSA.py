import random, math, time, itertools, joblib, scipy
from scipy.stats import norm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['figure.figsize'] = (10.0, 8.0) 
plt.rcParams['font.family'] = ['Times New Roman']
plt.rc('font', size=20)
style=itertools.cycle(["-","--","-.",":"])
e = math.e

def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

def Gaussian_Kernel(u):
    return math.exp(-1/2*u**2)/(math.sqrt(2*math.pi))

def Gaussian_Kernel_h(x, X, h):
    return Gaussian_Kernel((x-X)/h)/h

def I(x, v):
    return 1 if x <= v else 0

def scale(c, G):
    M = max(1, np.linalg.norm(G)/len(G)**.5)
    return c/M

def generate_delta():  
    u = random.uniform(0,1)
    if u<= 0.5:
        rst = 1
    else: 
        rst = -1
    return rst

def project_uni(val, ran):
    return max(min(val, ran[1]), ran[0])

def project(vec, space):
    d=len(vec)
    return np.array([project_uni(vec[i], space[i]) for i in range(d)])

# In all experiments, we consider $g(\theta,\lambda)=\sum_{j=1}^p\lambda_j$.
def KBSA(N, d,
         l0,v0,Gv0,Gl0,theta0,
         l_set,v_set,Gv_set,Gl_set,theta_set,
         a,b,bb,b1,bb1,c,h,
         q,m,generate_rv,evaluate,
         case=None,expect=False):
    l = [x[:] for x in l0]; v = v0[:]; Gv = Gv0[:]; Gl = [x[:] for x in Gl0]; theta=theta0[:]
    fs = [0]; RTs=[]
    sta=time.time()
    for i in range(N):  
        delta = np.array([generate_delta() for j in range(d)])
        c_ = scale(c(i), Gv[-1])      
        for p in range(len(m)): 
            c_ = scale(c_, Gl[p][-1])        
        Yp, Xp = generate_rv(theta[-1]+np.dot(c_, delta), case)
        Yn, Xn = generate_rv(theta[-1]-np.dot(c_, delta), case)
        Y, X = generate_rv(theta[-1], case)

        if expect==False:
            v_ = v[-1]+b(i)*q(theta[-1], X, v[-1])
            H = q(theta[-1], Xp, v[-1]+c_*np.inner(delta, Gv[-1]))\
                -q(theta[-1], Xn, v[-1]-c_*np.inner(delta, Gv[-1]))
            Gv_ = Gv[-1]+bb(i)*H/(2*c_)*delta

        theta_ = theta[-1]-a(i)*np.sum([Gl[i][-1] for i in range(len(m))],axis=0)
        
        Kp = Gaussian_Kernel_h(v[-1]+c_*np.inner(delta, Gv[-1]), Xp, h[1](i))
        Kn = Gaussian_Kernel_h(v[-1]-c_*np.inner(delta, Gv[-1]), Xn, h[1](i))

        for p in range(len(m)):
            l_ = l[p][-1]+b1[p](i)*Gaussian_Kernel_h(v[-1],X,h[0](i))*m[p](theta[-1],Y,  [l[j][-1] for j in range(p+1)])
            H = m[p](theta[-1], Yp, [l[j][-1]+c_*np.inner(delta, Gl[j][-1]) for j in range(p+1)])\
                -m[p](theta[-1], Yn, [l[j][-1]-c_*np.inner(delta, Gl[j][-1]) for j in range(p+1)])
            Gl_ = Gl[p][-1]+bb1[p](i)*Kp*Kn*H/(2*c_)*delta 
            l[p].append(project(l_,l_set))        
            Gl[p].append(project(Gl_,Gl_set))

        if expect==False:
            v.append(project(v_,v_set))        
            Gv.append(project(Gv_, Gv_set))
            
        theta.append(project(theta_, theta_set))
        fs.append(evaluate(theta_, case))

        if i==int(1e3)-1: RTs.append(time.time()-sta)
        if i==int(1e4)-1: RTs.append(time.time()-sta)
        if i==int(1e5)-1: RTs.append(time.time()-sta)
        if i==int(1e6)-1: RTs.append(time.time()-sta)
        
    return l, v, Gv, Gl, theta, fs, RTs

# 40 independent replications
def replication(d, l0,v0,Gv0,Gl0,theta0,
                l_set,v_set,Gv_set,Gl_set,theta_set,
                a,b,bb,b1,bb1,c,h,
                q,m,generate_rv,evaluate,
                case=None,expect=False):
    l1_lst3 = []; Gl1_lst3 = []; theta1_lst3 = []; fs1_lst3 = [];
    l1_lst4 = []; Gl1_lst4 = []; theta1_lst4 = []; fs1_lst4 = [];
    l1_lst5 = []; Gl1_lst5 = []; theta1_lst5 = []; fs1_lst5 = [];
    l1_lst6 = []; Gl1_lst6 = []; theta1_lst6 = []; fs1_lst6 = [];
    RT_lst = []
    r = 0; N=int(1e6)
    while True:
        with np.errstate(invalid='raise'):
            try:
                l1, v1, Gv1, Gl1, theta1, fs1, RT1 = KBSA(N, d,
                                                          l0,v0,Gv0,Gl0,theta0,
                                                          l_set,v_set,Gv_set,Gl_set,theta_set,
                                                          a,b,bb,b1,bb1,c,h,
                                                          q,m,generate_rv,evaluate,case,expect)
                l1_lst3.append([l[int(1e3)] for l in l1]); Gl1_lst3.append([l[int(1e3)] for l in Gl1]); 
                theta1_lst3.append(theta1[int(1e3)]);fs1_lst3.append(fs1[int(1e3)])
                l1_lst4.append([l[int(1e4)] for l in l1]); Gl1_lst4.append([l[int(1e4)] for l in Gl1]); 
                theta1_lst4.append(theta1[int(1e4)]);fs1_lst4.append(fs1[int(1e4)])
                l1_lst5.append([l[int(1e5)] for l in l1]); Gl1_lst5.append([l[int(1e5)] for l in Gl1]); 
                theta1_lst5.append(theta1[int(1e5)]);fs1_lst5.append(fs1[int(1e5)])
                l1_lst6.append([l[int(1e6)] for l in l1]); Gl1_lst6.append([l[int(1e6)] for l in Gl1]); 
                theta1_lst6.append(theta1[int(1e6)]);fs1_lst6.append(fs1[int(1e6)])
                RT_lst.append(RT1)
                r+=1
            except (FloatingPointError, ZeroDivisionError):
                print('Error: Division by Zero')
        if r>=40:
            break
    return l1_lst3, Gl1_lst3, theta1_lst3, fs1_lst3, \
           l1_lst4, Gl1_lst4, theta1_lst4, fs1_lst4, \
           l1_lst5, Gl1_lst5, theta1_lst5, fs1_lst5, \
           l1_lst6, Gl1_lst6, theta1_lst6, fs1_lst6, RT_lst