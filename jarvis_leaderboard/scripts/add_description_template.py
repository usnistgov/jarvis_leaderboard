import glob
x=[]
for i in glob.glob("../../docs/*/*/*.md"):
  if 'index' not in i:
     x.append(i)
     with open(i, "r") as file:
          filedata = file.read().splitlines()
     content=[]
     for jj,j in enumerate(filedata):
         if jj==0:
            j=j+'\n<!--benchmark_description-->\n'
            print(jj,j)   
         content.append(j)     
     with open(i, "w") as file:
        file.write("\n".join(content))
print(len(x))
