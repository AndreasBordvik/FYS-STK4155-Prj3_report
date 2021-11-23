# FYS-STK4155 Project 2

## Installation:

### Using conda:

Make sure that you are in the directory with the proj2_env.yml

```bash
$ conda env create -f environment_proj2.yml
```
```bash
$ conda activate environment_proj2
```

When done, deactive enviroment with:

```bash
$ conda deactivate
```

## Usage

The main report is found as a pdf, named as report/report.pdf, which is generated from the latex file main.tex. 

With enviroment activateded, running examples for each task can be found as jupyter notebooks in the code folder. This will reproduce the plots used in the report.
They can be run with the "Run all" command. 

All source code is found in common.py under the code folder. For studying figures and graphs without actually running the code, please refer to the figures folder. 

### Notes for running the notebooks:
   You may need to have latex locally installed in order to run using the fonts defined together with 
   the import statements. 
   ```sh
   plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
    "font.size": 10,
	})
   ```

Thanks!
