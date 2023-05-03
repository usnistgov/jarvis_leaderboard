import glob
for i in glob.glob('*/*/index.md'):
   print (i)
   content=[]
   f=open(i,'r')
   lines=f.read().splitlines()
   f.close()
   for ii in lines:
    tmp=ii
    if "<!--number_of_benchmarks--> - Number of benchmarks:" in ii:
          tmp=ii.replace("<!--number_of_benchmarks--> - Number of benchmarks:","<!--number_of_contributions--> - Number of contributions:")
   content.append(i)
   f=open(i,'w')
   f.write("\n".join(content))
   f.close()
   
