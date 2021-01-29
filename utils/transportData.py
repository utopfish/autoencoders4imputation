#@Time:2020/1/9 13:50
#@Author:liuAmon
#@e-mail:utopfish@163.com
#@File:transportData.py
__author__ = "liuAmon"
'''
对数据格式进行转换
'''
import os
import re
from collections import Counter
from utils.read_file import readNex
def count(str):
    '''
    算（）是否相同
    :param str:
    :return:
    '''
    count = Counter(str)
    print(count)

def readTre2mouse(path,speciesName,savePath):
    '''
    将数据从tnt格式转换为treespace能阅读模式
    :param path:
    :param speciesName:
    :param savePath:
    :return:
    '''
    res=[]
    with open(path, "r") as f:  # 打开文件
        data = f.read()
        for i in data.split('\n')[1:-1]:
            if 'proc-' in i:
                break
            i = i.replace(' )', ')')
            i= i.replace(' ',',')
            i= i.replace('*',';')
            i=i.replace(')(','),(')
            # count(i)
            for j in range(len(speciesName)-1,-1,-1):
                i = i.replace('{}'.format(j), '{}'.format(speciesName[j]))
            print(i)
            res.append(i)
    with open(savePath,'w') as f:
        for i in res:
            f.writelines(i+"\n")
def replace_char(old_string, char, index,old_len):
    '''
    字符串按索引位置替换字符
    '''
    old_string = str(old_string)
    # 新的字符串 = 老字符串[:要替换的索引位置] + 替换成的目标字符 + 老字符串[要替换的索引位置+1:]
    new_string = old_string[:index] + char + old_string[index+old_len:]
    return new_string
def hasP(index,i):
    """
    判断当前位置字符一直往做，是否会遇到P
    :param index:
    :param i:
    :return:
    """
    while (index>=0):
        if i[index]=="p":
            return False
        elif i[index]=="(":
            return True
        elif i[index]==",":
            return True
        index-=1
    return True

# #nex文件路径
# originFilePath=r'C:\Users\pro\Desktop\实验三蒙特卡洛树\真实数据集\Aria2015.nex'
# #tnt建树结果路径
# treFilePath=r'C:\Users\pro\Desktop\大论文实验1\simData.tre'
# treeSpaceFile=r'C:\Users\pro\Desktop\大论文实验1\simData.tnt'
#
# originData, miss_mask, speciesName, begin, end = readNex(originFilePath)
# simTre2mouse(treFilePath, speciesName, treeSpaceFile)
def simTre2mouse(path,speciesName,savePath):
    '''
    将模拟数据从tnt格式转换为treespace能阅读模式
    :param path:
    :param speciesName:
    :param savePath:
    :return:
    '''
    res=[]
    with open(path, "r") as f:  # 打开文件
        data = f.read()
        for i in data.split('\n')[1:-1]:
            if 'proc-' in i:
                break
            i = i.replace(' )', ')')
            i= i.replace(' ',',')
            i= i.replace('*',';')
            i=i.replace(')(','),(')
            # count(i)
            for j in range(len(speciesName) - 1, -1, -1):
                t=[loc.start() for loc in re.finditer(str(j), i)]
                for index in [loc.start() for loc in re.finditer(str(j), i)]:
                    if  hasP(index,i):
                        i=replace_char(i,'{}'.format(speciesName[j]),index,len(str(j)))
                        break
                #i = i.replace('{}'.format(j), '{}'.format(speciesName[j]),1)
            print(i)
            res.append(i)
    with open(savePath,'w') as f:
        for i in res:
            f.writelines(i+"\n")
def mouse2Tre(mouse,speciesName):
    '''
    将数据从tnt格式转换为treespace能阅读模式
    :param path:
    :param speciesName:
    :param savePath:
    :return:
    '''

    i = mouse.replace(')', ' )')
    i= i.replace(',',' ')
    i=i.replace('),(',')(')
    i = i.replace(';', '*')
    i = i.replace(') )', '))')
    i = i.replace(') )', '))')
    # count(i)
    for j in range(len(speciesName)-1,-1,-1):
        i = i.replace('{}'.format(speciesName[j]),'{}'.format(j))
    print(i)
def dirtyhand():
    path = r'C:\Users\pro\Desktop\int_data'
    # file='Aria2015.nex'
    # data, miss_row, speciesName = readNex(os.path.join(path, file))
    # mou='(Aysheaia,Anomalocaris,Hurdia,(Isoxys, Surusicaris,(((Canadaspis,Fuxianhuia),Occacaris),(Kunmingella,(((Martinssonia,Cephalocarida),Rehbachiella),((Agnostus,Kiisortoqia),((((Olenoides,Naraoia),Xandarella,Aglaspis),Emeraldella),((Jianfengia,Fortiforceps),(Yohoia,((((Leanchoilia_superlata,Leanchoilia_persephone),Leanchoilia_illecebrosa,(Oestokerkus,Yawunik),Actaeus,Oelandocaris),Alalcomenaeus),Haikoucaris)),((Offacolus,Dibasterium),(Weinbergina,Eurypterida))))))))));'
    # mouse2Tre(mou,speciesName)
    # mou2='(Aysheaia,(Anomalocaris ,Hurdia),(Isoxys,(((Canadaspis,Fuxianhuia),Occacaris),(Surusicaris,(Jianfengia,(Fortiforceps,((Yohoia,(((((Leanchoilia_superlata , Leanchoilia_persephone),Leanchoilia_illecebrosa,Actaeus,Oelandocaris),(Oestokerkus,Yawunik)),Alalcomenaeus),Haikoucaris)),((((Kunmingella,Agnostus),((Martinssonia,Cephalocarida),Rehbachiella)),(((Olenoides,Naraoia),Xandarella,Aglaspis),Emeraldella)),(((Offacolus,Dibasterium),(Weinbergina,Eurypterida)),Kiisortoqia)))))))))'
    # mouse2Tre(mou2, speciesName)
    file='Longrich2010.nex'
    data, miss_row, speciesName = readNex(os.path.join(path, file))
    mou='(Thescelosaurus_neglectus,Psittacosaurus_spp,((Stegoceras_validum,(Gravitholus_albertae,Colepiocephale_lambei)),Texacephale_langstoni,Hanssuesia_sternbergi,(Sphaerotholus_brevis,(Sphaerotholus_goodwini,(Sphaerotholus_edmontonense,Sphaerotholus_buchholtzae)),((Alaskacephale_gangloffi,Pachycephalosaurus_wyomingensis,(Stygimoloch_spinifer,Dracorex_hogwartsi)),(Tylocephale_gilmorei,Prenocephale_prenes,(Homalocephale_calathocercos,Goyocephale_lattimorei,Wannanosaurus_yansiensis))))));'
    mouse2Tre(mou,speciesName)
    file='Dikow2009.nex'
    data, miss_row, speciesName = readNex(os.path.join(path, file))
    mou='(Bombylius_major,((Apsilocephala_longistyla,(Prorates_sp_Escalante,(Phycus_frommeri,Hemigephyra_atra))),((Apiocera_painteri,((Opomydas_townsendi,Mydas_clavatus),(Mitrodetus_dentitarsis,(Nemomydas_brachyrhynchus,Afroleptomydas_sp_Clanwilliam)))),((Rhipidocephala_sp_HaroldJohnson,(Holcocephala_calva,Holcocephala_abdominalis)),((Perasis_transvaalensis,(Laphystia_tolandi,(Trichardis_effrena,(Nusa_infumata,((Laxenecera_albicincta,Hoplistomerus_nobilis),((Pilica_formidolosa,(Cerotainia_albipilosa,Atomosia_puella)),((Stiphrolamyra_angularis,Lamyra_gulo),(Laphria_aktis,Choerades_bella)))))))),((((Damalis_monochaetes,Damalis_annulata),(Rhabdogaster_pedion,Acnephalum_cylindricum)),(((Pegesimallus_laticornis,(Diogmites_grossus,(Plesiomma_sp_Guanacaste,(Dasypogon_diadema,(Saropogon_luteus,Lestomyia_fraudiger))))),((Trichoura_sp_Tierberg,Ablautus_coquilletti),(Molobratia_teutonus,(Nicocles_politus,(Leptarthrus_brevirostris,(Cyrtopogon_rattus,Ceraturgus_fasciatus)))))),((Willistonina_bilineata,(Eudioctria_albius,(Dioctria_hyalipennis,(Dioctria_rufipes,Dioctria_atricapillus)))),((Gonioscelis_ventralis,(Stenopogon_rufibarbis,Ospriocerus_aeacus)),((Tillobroma_punctipennis,(Prolepsis_tristis,Microstylum_sp_Karkloof)),(Lycostommyia_albifacies,(Scylaticus_costalis,Connomyia_varipennis))))))),(((Lasiopogon_cinctus,Lasiopogon_aldrichii),(Stichopogon_punctum,(Stichopogon_trifasciatus,Stichopogon_elegantulus))),(((Euscelidia_pulchra,Beameromyia_bifida),((Leptogaster_cylindrica,Leptogaster_arida),(Tipulogaster_glabrata,Lasiocnemus_lugens))),(((Emphysomera_pallidapex,Emphysomera_conopsoides),(Ommatius_tibialis,Afroestricus_chiastoneurus)),((Proctacanthus_philadelphicus,Pogonioefferia_pogonias),((Philodicus_tenuipes,(Promachus_amastrus,Megaphorus_pulchrus)),((Neolophonotus_bimaculatus,Dasophrys_crenulatus),(Neoitamus_cyanurus,(Clephydroneura_sp_Kepong,(Dysmachus_trigonus,(Philonicus_albiceps,(Machimus_occidentalis,(Tolmerus_atricapillus,(Asilus_sericeus,Asilus_crabroniformis)))))))))))))))))));'
    mouse2Tre(mou,speciesName)
    file='Liu2011.nex'
    data, miss_row, speciesName = readNex(os.path.join(path, file))
    mou='(Cycloneuralia,((Aysheaia,(Tardigrada,(Orstenotubulus,(Paucipodia,((Hadranax,Xenusion),(Microdictyon,(Cardiodictyon,(Hallucigenia,(Onychodictyon,(Luolishania,(Collins_monster,(Miraluolishania,Onychophora)))))))))))),(Jianshanopodia,(Megadictyon,(Kerygmachela,(Pambdelurion,(Opabinia,(((Anomalocaris,Laggania),Hurdia),(Diania,(Schinderhannes,(Fuxianhuia,(Leanchoilia,Euarthropoda))))))))))));'
    mouse2Tre(mou, speciesName)
def main():
    path = r'C:\Users\pro\Desktop\int_data'
    for file in os.listdir(path):
        try:
            if file.endswith('nex'):
                # file='Liu2011.nex'
                data, miss_row, speciesName = readNex(os.path.join(path, file))
                for ind, i in enumerate(speciesName):
                    print(ind, i)
                print((file[:-3] + 'tre'))
                treeSpecies = os.path.join(path, (file[:-3] + 'tre'))
                SvaetreeSpecies = os.path.join(path, (file[:-3] + 'txt'))
                readTre2mouse(treeSpecies, speciesName, SvaetreeSpecies)



                treeSpecies = os.path.join(path, file[:-4] + '_ii.tre')
                SvaetreeSpecies = os.path.join(path, file[:-4] + '_ii.txt')
                readTre2mouse(treeSpecies, speciesName, SvaetreeSpecies)

                treeSpecies = os.path.join(path, file[:-4] + '_knn.tre')
                SvaetreeSpecies = os.path.join(path, file[:-4] + '_knn.txt')
                readTre2mouse(treeSpecies, speciesName, SvaetreeSpecies)

                treeSpecies = os.path.join(path, file[:-4] + '_me.tre')
                SvaetreeSpecies = os.path.join(path, file[:-4] + '_me.txt')
                readTre2mouse(treeSpecies, speciesName, SvaetreeSpecies)

                treeSpecies = os.path.join(path, file[:-4] + '_sf.tre')
                SvaetreeSpecies = os.path.join(path, file[:-4] + '_sf.txt')
                readTre2mouse(treeSpecies, speciesName, SvaetreeSpecies)

                treeSpecies = os.path.join(path, file[:-4] + '_auto.tre')
                SvaetreeSpecies = os.path.join(path, file[:-4] + '_auto.txt')
                readTre2mouse(treeSpecies, speciesName, SvaetreeSpecies)

                treeSpecies = os.path.join(path, file[:-4] + '_newTech.tre')
                SvaetreeSpecies = os.path.join(path, file[:-4] + '_newTech.txt')
                readTre2mouse(treeSpecies, speciesName, SvaetreeSpecies)
        except Exception as e:
            print(e)

import pandas as pd
import os
import re
def bayes2Tre(path,savePath):
    # """
    # 将贝叶斯结果转化为treespace可读的树
    # :param path:Mr bayes输出文件.run1.t 路径
    # :param savePath:保存文件路径
    # :return:
    # sample：
    # path=r'C:\Users\pro\Desktop\bayes\0.5_03_Dikow2009.nex.run1.t'
    # savePath=r'C:\Users\pro\Desktop\Aguado2009_200树\0.5_03_Dikow2009.bayes.txt'
    # bayes2Tre(path,savePath)
    # """

    res = []
    speciesName={}
    #是否开始获取种名的标志
    flag=0
    count=0
    with open(path, "r") as f:  # 打开文件
        data = f.read()
        for i in data.split('\n')[1:-1]:
            if 'translate' in i:
                flag=1
                continue
            if flag==1 and "end;" not in i:

                if "[&U]" not in i :
                    tmp=i.replace(",","").replace(";","").split(" ")
                    speciesName[tmp[-2]]=tmp[-1]
                else:
                    i = re.sub(":[0123456789e\.\-\+]+", "", i.split("[&U] ")[1])
                    for j in range(len(speciesName), 0, -1):
                        i = i.replace('{}'.format(j), '{}'.format(speciesName[str(j)]))
                    res.append(i)
                    count+=1
                    print(count,i)
    with open(savePath,'w') as f:
        for i in res[:200]:
            f.writelines(i+"\n")


#将R模拟生成数据转化为nex数据
# 例子：
# path=r'C:\Users\pro\Desktop\ConstructTreeProject\sim.csv'
# savePath=r'C:\Users\pro\Desktop\ConstructTreeProject\sim.nex'
# sim2nex(path,savePath)
def sim2nex(path,savePath):
    """
    :param path:
    :param savePath:
    :return:
    """

    data=pd.read_csv(path,index_col=0)
    value=data.values
    indexs=data.index
    #注意修改分类单元数和特征数
    begin="#NEXUS \n begin data; \ndimensions ntax = {} nchar = {}; \nformat datatype = standard gap = - missing =? Interleave = no SYMBOLS = \"1234\";\nmatrix \n".format(len(value),len(value[0]))
    end=";\nEND;\n"
    with open(savePath,'w') as f:
        f.write(begin)
        for index,val in enumerate(value):
            t=[str(i) for i in val]
            f.writelines("{} ".format(indexs[index])+"".join(t)+"\n")
        f.write(end)
if __name__=="__main__":
    path=r'C:\Users\pro\Desktop\实验一缺失模式对建设的影响研究\模拟数据文件\test\50taxa25chara_sim.csv'
    savePath=r'C:\Users\pro\Desktop\实验一缺失模式对建设的影响研究\模拟数据文件\test\50taxa25chara_sim.nex'
    sim2nex(path,savePath)
