# Multigraph Analysis in Python

Ovaj projekat predstavlja rešenje trećeg domaćeg zadatka iz predmeta **Diskretne matematičke strukture**.  
Program je implementiran u programskom jeziku **Python** i služi za analizu neusmerenog multigrafa zadatog listom susedstva.

Graf se modeluje korišćenjem biblioteke **NetworkX**, dok se za određene proračune nad matricom susedstva koristi **NumPy** i linearna algebra.

## Functions

Program implementira analizu grafa i računa sledeće karakteristike:

- zbir stepena čvorova
- broj najkraćih puteva između dva čvora
- simetričnu razliku skupova suseda
- broj grana indukovanog podgrafa
- čvorove na određenoj udaljenosti u grafu
- ekscentricitet čvorova
- broj komponenti povezanosti nakon uklanjanja čvorova
- broj šetnji dužine *k* između čvorova pomoću matrice susedstva

Graf se tretira kao **neusmeren multigraf**, što znači da između dva čvora može postojati više paralelnih grana i da se one uzimaju u obzir u svim proračunima.

## Technologies

- Python  
- NetworkX  
- NumPy  
