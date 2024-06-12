from tabula import read_pdf
#from tabulate import tabulate
 
#reads table from pdf file
df = read_pdf("Amazon_2023_10K.pdf",pages="all") #address of pdf file

dfs = tabula.read_pdf(pdf_path, pages=37, stream=True)
print(dfs)

#print(tabulate(df))