# A kernel based stochastic approximation framework for contextual optimization

This archive is distributed under the [MIT License](LICENSE).

The software and data in this repository are a snapshot of the software
that was used in the research reported on in the paper 
[A kernel-based stochastic approximation framework for contextual optimization](https://papers.ssrn.com/abstract=5411562) by Hao Cao, Jian-Qiang Hu, and Jiaqiao Hu. 

## Cite

To cite the contents of this repository, please cite the paper.
```
@misc{cao2024kernel,
  author =        {Hao Cao and Jian-Qiang Hu and Jiaqiao Hu},
  journal =       {SSRN Electronic Journal},
  title =         {A kernel based stochastic approximation framework for contextual optimization},
  year =          {2024},
  note =          {Available at SSRN: https://ssrn.com/abstract=5411562},
  url  =          {https://ssrn.com/abstract=5411562},
}  
```

## Description

The goal of this software is to demonstrate a kernel-based stochastic approximation (KBSA) 
framework for solving contextual stochastic optimization problems with differentiable objective functions, 
including conditional expectations, conditional quantiles, and many co-risk measures such as CoVaR and CoES. 
This unifying algorithmic framework can not only be applied for estimating multiple contextual measures
but also allows their sensitivity analysis and optimization. 

## Structure

The structure of this repository is as follows:
- `src/KBSA.py`: The code of the proposed kernel-based stochastic approximation (KBSA) algorithm.
- `scripts/`: The directory containing implementation examples.
- `results/`: The directory storing the figures included in the paper.

## Requirements
The code is tested in the environment of Python 3.12.4 with Windows 11.  

Numerous package requirements exist, including but not limited to: 
`numpy`, `pandas`, `matplotlib`, `scipy`, `operator`, `IPython`, 
`random`, `itertools`, `math`, `pickle`, `joblib`, `seaborn`, and `statsmodels`. 
For configuration details, please refer to the conda environment manifest file `scripts/environment.yml`.

## Results
We provide detailed implementation guidances and demonstrate results 
in the `scripts/` directory. It suffices just to run each notebook (you can execute cells individually). 
We have generated all the data used in the paper——no external data sources are required. 
- `KBSA_test_mean.ipynb`: The Python notebook containing implementation guidances
for reproducing the results in the **blackbox test functions** examples with **Cost 1**;
- `KBSA_test_risk.ipynb`: The Python notebook containing implementation guidances
for reproducing the results in the **blackbox test functions** examples with **Cost 2**;
- `KBSA_portfolio.ipynb`: The Python notebook containing implementation guidances
for reproducing the results in the **nonlinear portfolios** example.
