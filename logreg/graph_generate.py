x1=[]
y1=[]
x2=[]
y2=[]
x3=[]
y3=[]
flag=0
with open("graph_plot.txt") as fh:
    text=fh.read().splitlines()
    for line in text:
        if line=="eta=0.1":
            flag=0.1
        elif line=="eta=0.2":
            flag=0.2
        elif line=="eta=0.3":
            flag=0.3
        elif flag==0.1:
            x,y=line.split(":")
            x1.append(x)
            y1.append(y)
        elif flag==0.2:
            x,y=line.split(":")
            x2.append(x)
            y2.append(y)
        elif flag==0.3:
            x,y=line.split(":")
            x3.append(x)
            y3.append(y)
        else:
            continue
