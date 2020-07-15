import matplotlib.pyplot as plt
import numpy as np
def as_num(x):
     y='{:.5f}'.format(x) # 5f表示保留5位小数点的float型
     return(y)
if __name__=="__main__":
    with open(r'G:\labWork\Autoencoders_Interpolation\logs\2020-07-05-00h.log') as file_obj:
        content = file_obj.read()
        tem=content.split("INFO     |")
        resu=[]
        temp={}
        temp['miss'] = []
        temp['data'] = []
        for i in content.split("INFO     |"):
            if "247 -" in i :
                if temp != {}:
                    resu.append(temp)
                    temp = {}
                    temp['miss'] = []
                    temp['data'] = []
                temp['name']=i.split("247 -")[1].split("\n")[0].replace("*","")
            if "334 -" in i:
                temp['chara']=i.split("334 -")[1].split("\n")[0]
            if "335 -" in i:
                temp['miss'].append(i.split("335 -")[1].split("\n")[0].replace("missing rate is",""))
            if "<module>:336" in i :
                t=i.split("[")[1].split("]")[0].split(" ")
                while '' in t:
                    t.remove('')
                t=[float(i.replace("\n","")) for i in t]

                temp['data'].append(t)
        print(resu[0])
        color=['blue','green','red','yellow','black','burlywood','cadetblue','chartreuse','chocolate','coral']
        for i in resu[1:]:
            plt.figure()
            index=0
            for miss_rate,y in zip(i['miss'],i['data']):
                x = np.linspace(0, 6, 12)
                plt.plot(x,y,color=color[index],label='{}'.format(miss_rate))
                index+=1
            plt.title("{}:{}".format(i['name'],i['chara']))
            plt.legend(loc="upper left")
            plt.show()
                # print(t)
                # print("-----")
                # x=np.linspace(0,6,12)
                # plt.plot(x, t)
                # plt.show()
