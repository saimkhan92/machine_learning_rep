from matplotlib import pyplot as plt
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
            x1.append(float(x))
            y1.append(float(y))
        elif flag==0.2:
            x,y=line.split(":")
            x2.append(float(x))
            y2.append(float(y))
        elif flag==0.3:
            x,y=line.split(":")
            x3.append(float(x))
            y3.append(float(y))
        else:
            continue

plt.plot(x1,y1,linewidth=1.5)
plt.plot(x2,y2,linewidth=1.5)
plt.plot(x3,y3,linewidth=1.5)


plt.title('Learning rate affect on convergence')
plt.ylabel('ho_acc')
plt.xlabel('iteration')

plt.show()
