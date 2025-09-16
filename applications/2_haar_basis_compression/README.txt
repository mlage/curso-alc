Demonstração da compressão de imagens usando bases de Haar (wavelets)

1) Gere os arquivos binários com as matrizes da mudança de base. Para isso, basta rodar o script gen_basis_wavelets.py

2) Compile a lib em C responsável por fazer a compressão e decompressão: gcc -shared -fPIC -o img_compression.so img_compression.c -fopenmp -O3

3) Rode o script exemplo: compress.py

IMPORTANTE: O exemplo usa como input um campo escalar de floats sobre um grid. Na prática, uma imagem. Pode ser adaptado para ser mais eficiente com RGB uint8.
