import h2o
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.metrics import precision_recall_curve
import pickle 

def calculate_efficiency_pool_thr(probs_one,groundtruth,poolsize,threshold):

    tests=0
    predicted_positives = 0
    real_positives = 0
    predicted_thr = (probs_one>threshold).astype(int)
    tests += predicted_thr.sum()
    detected_idx = np.where(predicted_thr==1)[0]
    predicted_positives += groundtruth[detected_idx].sum()
    real_positives += groundtruth[detected_idx].sum()
    complete_results = np.concatenate((probs_one.reshape(-1,1),groundtruth.reshape(-1,1)),axis=1)
    complete_results = np.delete(complete_results,detected_idx,axis=0)
    sorted_results = complete_results[np.argsort(complete_results[:, 0])[::-1]]
    groups = int(np.ceil(len(sorted_results)/poolsize))

    for g in range(groups):
        current = sorted_results[g*poolsize:poolsize+(poolsize*g)]
        tests +=1
        current_positives = current[:,1].sum()
        if current_positives > 0:
            tests += len(current)
            predicted_positives += current_positives

    efficiency = len(probs_one)/tests

    #print('Smart Pooling efficiency is '+ str(efficiency))

    return efficiency

def calculate_efficiency_best(probs_one,groundtruth):
    
    best_thr = []
    best_eff = []

    thresholds = np.arange(0,1,0.0005)
    pools = np.arange(2,13)
    for poolsize in tqdm(pools):
        efficiency_thr = []
        for thr in thresholds:
            tests=0
            predicted_positives = 0
            predicted_thr = (probs_one>thr).astype(int)
            tests += predicted_thr.sum()
            detected_idx = np.where(predicted_thr==1)[0]
            predicted_positives += groundtruth[detected_idx].sum()
            complete_results = np.concatenate((probs_one.reshape(-1,1),groundtruth.reshape(-1,1)),axis=1)
            complete_results = np.delete(complete_results,detected_idx,axis=0)
            sorted_results = complete_results[np.argsort(complete_results[:, 0])[::-1]]
            groups = int(np.ceil(len(sorted_results)/poolsize))

            for g in range(groups):
                current = sorted_results[g*poolsize:poolsize+(poolsize*g)]
                tests +=1
                current_positives = current[:,1].sum()
                if current_positives > 0:
                    tests += len(current)
                    predicted_positives += current_positives

            efficiency = len(probs_one)/tests
            efficiency_thr.append(efficiency)
        max_id_thr = np.argmax(efficiency_thr)
        best_thr.append(thresholds[max_id_thr])
        best_eff.append(efficiency_thr[max_id_thr])
    max_id_pool = np.argmax(best_eff)
    #print('Smart Pooling efficiency is '+ str(best_eff[max_id_pool]))

    return best_eff[max_id_pool],pools[max_id_pool],best_thr[max_id_pool]

def calculate_efficiency(probs_one,groundtruth,ventana):
    area = 0
    todos_tests = []
    todos_eficiencia = []
    R_total = []
    P_total = []
    F1_total = []
    F2_total = []
    todos_buenas = []

    umbrales = np.arange(0,1,0.0005)
    for thr in tqdm(umbrales):
        tests=0
        pacientes = 0
        buenas = 0
        predicted_thr = (probs_one>thr).astype(int)
        tests += predicted_thr.sum()
        indices_detectados = np.where(predicted_thr==1)[0]
        pacientes += groundtruth[indices_detectados].sum()
        buenas += groundtruth[indices_detectados].sum()
        complete_results = np.concatenate((probs_one.reshape(-1,1),groundtruth.reshape(-1,1)),axis=1)#np.concatenate((probs_one,groundtruth),axis=1)
        complete_results = np.delete(complete_results,indices_detectados,axis=0)
        sorted_results = complete_results[np.argsort(complete_results[:, 0])[::-1]]
        groups = int(np.ceil(len(sorted_results)/ventana))

        for g in range(groups):
            actuales = sorted_results[g*ventana:ventana+(ventana*g)]
            tests +=1
            positivos = actuales[:,1].sum()
            if positivos > 0:
                tests += len(actuales)
                pacientes += positivos

        eficiencia = len(probs_one)/tests
        todos_tests.append(tests)
        todos_eficiencia.append(eficiencia)
        todos_buenas.append(buenas)

        TP = np.sum((predicted_thr == 1) & (groundtruth ==1))
        FN = np.sum((predicted_thr == 0) & (groundtruth ==1))
        FP = np.sum((predicted_thr == 1) & (groundtruth ==0))
        Recall = TP/(TP+FN)
        if TP == 0 & FP ==0:
            Precision = 0
        else:
            Precision = TP/(TP+FP)
        F1= 2*(Precision*Recall)/(Precision+Recall)
        F2= 5*(Precision*Recall)/(4*Precision+Recall)
        
        R_total.append(Recall)
        P_total.append(Precision)
        F1_total.append(F1)
        F2_total.append(F2)

    max_eficiencia = np.argmax(todos_eficiencia)
    for i,esto in enumerate(zip(np.array(R_total)[:-1],np.array(R_total)[1:])): 
            area+=(np.abs(esto[1]-esto[0]))*P_total[i]
    print('The area is '+ str(area))
    print('Max Efficiency is '+ str(todos_eficiencia[max_eficiencia]))
    print('Threshold is '+ str(umbrales[max_eficiencia]))
    print('F1 is '+ str(F1_total[max_eficiencia]))
    print('F2 is '+ str(F2_total[max_eficiencia]))
    print('Precision is '+ str(P_total[max_eficiencia]))
    print('Recall is '+ str(R_total[max_eficiencia]))
    print('Good are '+ str(todos_buenas[max_eficiencia]))

    return umbrales, todos_eficiencia

def random_efficiency(gt,ventana):
    todos_tests = []
    todos_eficiencia = []
    ids = np.arange(len(gt))

    for i in range(2000):
        tests=0
        pacientes = 0
        random.shuffle(ids)
        gt=gt[ids]
        groups = int(np.ceil(len(gt)/ventana))

        for g in range(groups):
            actuales = gt[g*ventana:ventana+(ventana*g)]
            tests +=1
            positivos = actuales.sum()
            if positivos > 0:
                tests += len(actuales)
                pacientes += positivos

        eficiencia = len(gt)/tests
        todos_tests.append(tests)
        todos_eficiencia.append(eficiencia)

    #print('Dorfman Efficiency is '+ str(np.mean(todos_eficiencia)))
    #print('Std Efficiency is '+ str(np.std(todos_eficiencia)))
    return np.mean(todos_eficiencia)

def max_efficiency(gt,poolsize):
    original=gt
    indices_detectados = np.where(gt==1)[0]
    tests = gt[indices_detectados].sum()
    gt = np.delete(gt,indices_detectados,axis=0)
    groups = int(np.ceil(len(gt)/poolsize))
    tests += groups
    eficiencia = len(original)/tests
    print('Max Efficiency is '+ str(eficiencia))

    return eficiencia

def prevalence_efficiency(df,filterdate,uniandes,ventana=7):
    trainval = df[df['date']<filterdate].copy()
    test = df[df['date']>filterdate].copy()
    test = h2o.H2OFrame(test, column_names=list(test.columns.astype(str)))
    test= test.as_data_frame()

    train_last=trainval.drop_duplicates(subset='test_center',keep='last')
    p_keys=train_last['test_center'].unique()
    p_values=train_last['positives_accum']/train_last['tests_accum']
    dictionary = dict(zip(p_keys, p_values))
    test['prev_acum']=test['test_center']
    test['prev_acum'].replace(dictionary,inplace=True)
    merged = pd.merge(test,uniandes,on=['date','test_center'])

    pooling_list= np.array(merged[['prev_acum','result']])
    pooling_list = pooling_list[np.argsort(pooling_list[:, 0])[::-1]]
    tests=0
    positivos=0
    groups = int(np.ceil(len(pooling_list)/ventana))
    for g in range(groups):
        actuales = pooling_list[g*ventana:ventana+(ventana*g)]
        tests +=1
        positivos = actuales[:,1].sum()
        if positivos > 0:
            tests += len(actuales)
    eficiencia = len(pooling_list)/tests
    
    return eficiencia

def metrics(merged,graph_name,prev_eff=-1,window=7,savegraph=False):
    probability = merged['pred_incidence'].values
    groundtruth = merged['result'].values
    prevalence =100*(groundtruth.sum()/len(groundtruth))

    print('calculating efficiency on prevalence: {}'.format(prevalence))
    
    thresholds, efficiency = calculate_efficiency(probability,groundtruth,window)
    random_eff = random_efficiency(groundtruth,window)
    maximum = max_efficiency(groundtruth,window)
    random_eff=random_eff.repeat(len(thresholds))
    maximum=np.ones(len(thresholds))*maximum
    pre,re,th = precision_recall_curve(groundtruth.astype('int'),probability)

    if prev_eff !=-1:
        print('prevalence_pooling: {}'.format(prev_eff))
        plot_prev_eff = np.ones(len(thresholds))*prev_eff
    
    #
    if savegraph:
        fig = plt.figure() 
        ax = plt.subplot(111) 
         
        ax.plot(thresholds, efficiency, label = "Smart pooling:{}".format(round(np.max(efficiency),2))) 
        ax.plot(thresholds, maximum, label = "Upper bound:{}".format(round(maximum[0],2)),linestyle='dashdot')  
        ax.plot(thresholds, random_eff, label = "Standard pooling:{}".format(round(random_eff[0],2)),linestyle='dashed')
        if prev_eff !=-1: 
            ax.plot(thresholds, plot_prev_eff, label = "Prevalence pooling:{}".format(round(prev_eff,2)),linestyle='dashed') 

        plt.xlabel('Threshold') 
        plt.ylabel('Efficiency') 
        plt.title('Efficiency Robustness on {}% prevalence, pool={}'.format(round(prevalence,2),window)) 
         
        box = ax.get_position() 
        ax.set_position([box.x0, box.y0 + box.height * 0.1, 
                         box.width, box.height * 0.9]) 
        fontP = FontProperties() 
        fontP.set_size('small') 
        ax.legend(loc='upper center', bbox_to_anchor=(0.49, -0.11), 
                  fancybox=True, shadow=True, ncol=3,prop=fontP) 
        fig.savefig(graph_name)

        fig = plt.figure() 
        ax = plt.subplot(111) 
         
        ax.plot(thresholds, efficiency, label = "Smart pooling:{}".format(round(np.max(efficiency),2))) 
        ax.plot(thresholds, random_eff, label = "Standard pooling:{}".format(round(random_eff[0],2)),linestyle='dashed')
        if prev_eff !=-1: 
            ax.plot(thresholds, plot_prev_eff, label = "Prevalence pooling:{}".format(round(prev_eff,2)),linestyle='dashed')

        plt.xlabel('Threshold') 
        plt.ylabel('Efficiency') 
        plt.title('Efficiency Robustness on {}% prevalence, pool={}'.format(round(prevalence,2),window)) 
         
        box = ax.get_position() 
        ax.set_position([box.x0, box.y0 + box.height * 0.1, 
                         box.width, box.height * 0.9]) 
        fontP = FontProperties() 
        fontP.set_size('small') 
        ax.legend(loc='upper center', bbox_to_anchor=(0.49, -0.11), 
                  fancybox=True, shadow=True, ncol=3,prop=fontP) 
        fig.savefig(graph_name.replace('.pdf','-nomax.pdf'))

        fig = plt.figure() 
        ax = plt.subplot(111) 
        ax.plot(re,pre)
        plt.xlabel('Recall') 
        plt.ylabel('Precision') 
        plt.title('PR curve') 
        fig.savefig(graph_name.replace('.pdf','-pr.pdf'))

        pickle_out = open(graph_name.replace('.pdf','') + "Real_Data.pickle","wb")
        pickle.dump([thresholds, random_eff, efficiency, maximum], pickle_out)
        pickle_out.close()
        pickle_out2 = open(graph_name.replace('.pdf','') + "PR_Data.pickle","wb")
        pickle.dump([re, pre], pickle_out2 )
        pickle_out2.close()
    return prevalence, efficiency, random_eff, maximum



def evaluate_prev_acum(model_leader,test,df,filter_date,calc_preveff,test_route='datos/uniandes/Uniandes_tanda2.xlsx',window=7,graph_name='results.pdf'):
    preds =model_leader.predict(test)
    #
    predictions=preds[preds.columns[0]].as_data_frame().values
    df_test = test.as_data_frame().copy()
    df_test['pred_positive'] = predictions.astype('int')

    uniandes = pd.read_excel(test_route)
    uniandes = h2o.H2OFrame(uniandes, column_names=list(uniandes.columns.astype(str)))
    uniandes= uniandes.as_data_frame()
    uniandes['Institución']=uniandes['Institución'].replace(' ','',regex=True)
    uniandes.rename(columns={'Fecha recepción':'date','Institución':'test_center','Resultado':'result'},inplace=True)
    uniandes=uniandes[uniandes['result']!='Inválido']
    uniandes.loc[:,'result']=uniandes['result'].replace('positivo','Positivo',regex=True)
    uniandes.loc[uniandes['result']=='Negativo','result']=0
    uniandes.loc[uniandes['result']=='Positivo','result']=1
    uniandes=uniandes.loc[uniandes['result']!='Indeterminado',:]
    uniandes.reset_index(inplace = True, drop = True) 
    if calc_preveff:
        prev_eff = prevalence_efficiency(df,filter_date,uniandes,window)
    else:
        prev_eff= -1

    df = h2o.H2OFrame(df, column_names=list(df.columns.astype(str)))
    df = df.as_data_frame()
    df = df[['date','test_center','positives_accum']]

    last_train_df = pd.merge(df,df_test,on=['date','test_center'])
    last_train_df = last_train_df.drop_duplicates(subset='test_center',keep='first')
    last_train_df = last_train_df[['date','test_center','positives_accum']]
    merged_test = pd.merge(last_train_df,df_test,on=['date','test_center'],how='outer')
    merged_test.sort_values(by=['test_center','date'],inplace=True)
    merged_test.fillna(value=-1,inplace=True)
    merged_test['update_positives']=np.nan
    merged_test.reset_index(inplace = True, drop = True) 
    
    for index, row in merged_test.iterrows():
        if row['positives_accum'] != -1:
            merged_test.loc[index,'update_positives'] = row['positives_accum']+row['pred_positive']
        else:
            merged_test.loc[index,'positives_accum'] = merged_test.loc[index-1,'update_positives']
            merged_test.loc[index,'update_positives'] = merged_test.loc[index,'positives_accum']+row['pred_positive']

    merged_test['pred_incidence'] = merged_test['update_positives']/merged_test['tests_accum']


    merged = pd.merge(merged_test,uniandes,on=['date','test_center'])
    metrics(merged,graph_name,prev_eff,window)


def evaluate(model_leader,test,df,filter_date,calc_preveff,savegraph,test_route='datos/uniandes/Uniandes_tanda2.xlsx',window=7,graph_name='results.pdf'):
    preds =model_leader.predict(test)
    predictions=preds[preds.columns[0]].as_data_frame().values
    df_test = test.as_data_frame().copy()
    # df_test['pred_positive_accum'] = predictions[:,0].astype('float') * df_test['tests_accum']
    # df_test['positive_diff'] =  df_test['pred_positive_accum'] - df_test['lag_pos']
    # df_test['pred_incidence'] = predictions.astype('float') * df_test['tests_accum']
    df_test['pred_incidence'] = predictions.astype('float').squeeze()

    uniandes = pd.read_excel(test_route)
    uniandes = h2o.H2OFrame(uniandes, column_names=list(uniandes.columns.astype(str)))
    uniandes= uniandes.as_data_frame()
    uniandes['Institución']=uniandes['Institución'].replace(' ','',regex=True)
    uniandes.rename(columns={'Fecha recepción':'date','Institución':'test_center','Resultado':'result'},inplace=True)
    uniandes=uniandes[uniandes['result']!='Inválido']
    uniandes.loc[:,'result']=uniandes['result'].replace('positivo','Positivo',regex=True)
    uniandes.loc[uniandes['result']=='Negativo','result']=0
    uniandes.loc[uniandes['result']=='Positivo','result']=1
    uniandes=uniandes.loc[uniandes['result']!='Indeterminado',:]
    uniandes.reset_index(inplace = True, drop = True) 

    if calc_preveff:
        prev_eff = prevalence_efficiency(df,filter_date,uniandes,window)
    else:
        prev_eff= -1
    merged = pd.merge(df_test,uniandes,on=['date','test_center'])
    prevalence, efficiency, random_eff, maximum = metrics(merged,graph_name,prev_eff,window,savegraph)
    return prevalence, efficiency, random_eff, maximum

def evaluate_best(model_leader,test,df, test_route='data/TestCenter.xlsx'):
    preds =model_leader.predict(test)
    predictions=preds[preds.columns[0]].as_data_frame().values
    df_test = test.as_data_frame().copy()
    df_test['pred_incidence'] = predictions.astype('float').squeeze()

    uniandes = pd.read_excel(test_route)
    uniandes = h2o.H2OFrame(uniandes, column_names=list(uniandes.columns.astype(str)))
    uniandes= uniandes.as_data_frame()
    uniandes['Institución']=uniandes['Institución'].replace(' ','',regex=True)
    uniandes.rename(columns={'Fecha recepción':'date','Institución':'test_center','Resultado':'result'},inplace=True)
    uniandes=uniandes[uniandes['result']!='Inválido']
    uniandes.loc[:,'result']=uniandes['result'].replace('positivo','Positivo',regex=True)
    uniandes.loc[uniandes['result']=='Negativo','result']=0
    uniandes.loc[uniandes['result']=='Positivo','result']=1
    uniandes=uniandes.loc[uniandes['result']!='Indeterminado',:]
    uniandes=uniandes.loc[~uniandes.result.isna()]
    uniandes.reset_index(inplace = True, drop = True) 
    merged = pd.merge(df_test,uniandes,on=['date','test_center'])

    probability = merged['pred_incidence'].values
    groundtruth = merged['result'].values
    prevalence =100*(groundtruth.sum()/len(groundtruth))

    print('calculating efficiency on prevalence: {}'.format(prevalence))
    
    efficiency,poolsize,threshold = calculate_efficiency_best(probability,groundtruth)
    random_eff = random_efficiency(groundtruth,poolsize)
    maximum = max_efficiency(groundtruth,poolsize)
    print('Best in test - thr:{}, pool:{}, eff:{}'.format(threshold,poolsize,efficiency))

    return prevalence, efficiency, random_eff, maximum,threshold, poolsize

def get_from_val(val_model,val,df,test_route='data/TestCenter.xlsx'):
    preds =val_model.predict(val)
    predictions=preds[preds.columns[0]].as_data_frame().values
    df_val = val.as_data_frame().copy()
    df_val['pred_incidence'] = predictions.astype('float').squeeze()

    uniandes = pd.read_excel(test_route)
    uniandes = h2o.H2OFrame(uniandes, column_names=list(uniandes.columns.astype(str)))
    uniandes= uniandes.as_data_frame()
    uniandes['Institución']=uniandes['Institución'].replace(' ','',regex=True)
    uniandes.rename(columns={'Fecha recepción':'date','Institución':'test_center','Resultado':'result'},inplace=True)
    uniandes=uniandes[uniandes['result']!='Inválido']
    uniandes.loc[:,'result']=uniandes['result'].replace('positivo','Positivo',regex=True)
    uniandes.loc[uniandes['result']=='Negativo','result']=0
    uniandes.loc[uniandes['result']=='Positivo','result']=1
    uniandes=uniandes.loc[uniandes['result']!='Indeterminado',:]
    uniandes=uniandes.loc[~uniandes.result.isna()]
    uniandes.reset_index(inplace = True, drop = True) 

    merged = pd.merge(df_val,uniandes,on=['date','test_center'])
    probability = merged['pred_incidence'].values
    groundtruth = merged['result'].values
    prevalence =100*(groundtruth.sum()/len(groundtruth))
    print('calculating val efficiency on prevalence: {}'.format(prevalence))
    val_eff,poolsize,threshold = calculate_efficiency_best(probability,groundtruth)
    print('Validation - thr: {}, pool: {}, eff:{}'.format(threshold,poolsize,val_eff))

    return threshold,poolsize

def evaluate_thr_pool(model_leader,test,df,threshold, poolsize, test_route='data/TestCenter.xlsx'):
    preds =model_leader.predict(test)
    predictions=preds[preds.columns[0]].as_data_frame().values
    df_test = test.as_data_frame().copy()
    df_test['pred_incidence'] = predictions.astype('float').squeeze()

    uniandes = pd.read_excel(test_route)
    uniandes = h2o.H2OFrame(uniandes, column_names=list(uniandes.columns.astype(str)))
    uniandes= uniandes.as_data_frame()
    uniandes['Institución']=uniandes['Institución'].replace(' ','',regex=True)
    uniandes.rename(columns={'Fecha recepción':'date','Institución':'test_center','Resultado':'result'},inplace=True)
    uniandes=uniandes[uniandes['result']!='Inválido']
    uniandes.loc[:,'result']=uniandes['result'].replace('positivo','Positivo',regex=True)
    uniandes.loc[uniandes['result']=='Negativo','result']=0
    uniandes.loc[uniandes['result']=='Positivo','result']=1
    uniandes=uniandes.loc[uniandes['result']!='Indeterminado',:]
    uniandes=uniandes.loc[~uniandes.result.isna()]
    uniandes.reset_index(inplace = True, drop = True) 
    merged = pd.merge(df_test,uniandes,on=['date','test_center'])

    probability = merged['pred_incidence'].values
    groundtruth = merged['result'].values
    prevalence =100*(groundtruth.sum()/len(groundtruth))

    print('calculating efficiency on prevalence: {}'.format(prevalence))
    
    efficiency = calculate_efficiency_pool_thr(probability,groundtruth,poolsize,threshold)
    random_eff = random_efficiency(groundtruth,poolsize)
    print('Test - thr: {}, pool: {}, eff:{}'.format(threshold,poolsize,efficiency))
    return prevalence, efficiency, random_eff
