import IPython.nbformat.current as nbf
nb = nbf.read(open(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\HSI Classification using CNN\SOM_v0.py', 'r'), 'py')
nbf.write(nb, open(r'C:\Users\user\Desktop\Abhilash\Imp\CEERI\NN\HSI Classification using CNN\SOM_v0.ipynb', 'w'), 'ipynb')