/*##############################################################################
## HPCC SYSTEMS software Copyright (C) 2023 HPCC Systems®.  All rights reserved.
#################################################   ############################# */
Import Python3 as PYTHON;
IMPORT $;
IMPORT ML_Core;
IMPORT ML_Core.Types as CTypes;
NumericField := CTypes.NumericField;
DiscreteField := CTypes.DiscreteField;


/* The RANDOMFOREST() is used to obtain the set of list of prediction of both Classification and Regression types using the DecisionTree function that uses the python functions using EMBED
@class_prediction is a set of strings that contain the classification predictions of the individual decision trees
@reg_prediction is a set of real numbers that contain the regression predictions of the individual decision trees
@tree_model is a set of strings that is use to represent the decision tree model used to make predictions
*/

EXPORT RandomForest():=MODULE
    SHARED class_prediction:=RECORD
        set of integer prediction;
    END;
    SHARED class_prediction_id:=RECORD
        UNSIGNED id;
        set of integer prediction;
    END;
    SHARED reg_prediction:=RECORD
        set of real prediction;
    END;
    SHARED reg_prediction_id:=RECORD
        UNSIGNED id;
        set of real prediction;
    END;
    SHARED tree_model:=RECORD
        string model;
    END;

    /*Random Forest function for Classification using sklearn library of python using the                   RandomForestClassifier() function.*/    
    /*@INDEPENDENTS is the part of the input dataset on which the prediction depends
    *@DEPENDENTS is the part of the dataset which can be predicted based on the INDEPENDENTS columns and they form a part of the prediction result of the Decison Tree
    *@nthreads decides the number of threads used by the processor to execute the RandomForestClassifier algorithm on the same dataset across multiple nodes of the cluster.
    *@ntrees decides the number of times the RandomForestClassifier algorithms that is executed by the processor
    
    *@ntrees_perthread is the number of DecisionTree models that are executed by each thread of the process.
    *@last_thread is the number of DecisionTrees left for the last thread after symmetric distribution of the trees  
    */
    EXPORT STREAMED DATASET(class_prediction) RAND_CLASSTREE(DATASET(NumericField)
    independents, DATASET(DiscreteField) dependents,INTEGER nthreads=1,INTEGER ntrees=1,
    UNSIGNED maxDepth=100):=EMBED
    (PYTHON : globalscope('globalscope'),persist('query'),activity)
        import pandas as pd
        from sklearn import tree
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.tree import export_text
        import threading
        import math
        global class_dectrees
        class_dectrees=[]
        threads=[]
        independents=list(independents)
        dependents=list(dependents)
        outRecs = []
        fields = []
        prevId = 0
        for item in independents:
            wi, id, number, value = item

            if id != prevId :
                if prevId > 0:
                    outRecs.append(fields)
                    fields = []

                prevId = id
            fields.append(value)
        X=pd.DataFrame(outRecs)

        outRecs = []
        fields = []
        prevId = 0
        for item in dependents:
            wi, id, number, value = item

            if id != prevId :
                if prevId > 0:
                    outRecs.append(fields)
                    fields = []

                prevId = id
            fields.append(value)
        Y=pd.DataFrame(outRecs)
        
     
        last_thread=ntrees%nthreads
        ntrees_perthread=math.floor(ntrees/nthreads)
        if(last_thread!=0):
            ntrees_perthread+=1
            last_thread=ntrees-(ntrees_perthread*(nthreads-1))
        def Dec_tree_class(nt):
            dec_tree_class=RandomForestClassifier(n_estimators=nt,max_depth=maxDepth)
            dec_tree_class.fit(X,Y)                
            class_dectrees.append(dec_tree_class)
               
        for i in range(nthreads):
            if((nthreads-i)==1):
                t=threading.Thread(target=Dec_tree_class,args=(last_thread,))
                t.start()
                threads.append(t)
            else:
                t=threading.Thread(target=Dec_tree_class,args=(ntrees_perthread,))
                t.start()
                threads.append(t)

        for t in range(nthreads):
            threads[t].join()
   
        return []
    ENDEMBED;



    /*Random Forest function for Regression using sklearn library of python using the                   RandomForestRegressor() function.*/    
    /*@INDEPENDENTS is the part of the input dataset on which the prediction depends
    *@DEPENDENTS is the part of the dataset which can be predicted based on the INDEPENDENTS columns and they form a part of the prediction result of the Decison Tree
    *@nthreads decides the number of threads used by the processor to execute the RandomForestRegressoralgorithm on the same dataset across multiple nodes of the cluster.
    *@ntrees decides the number of times the RandomForestRegressor algorithms that is executed by the processor
    
    *@ntrees_perthread is the number of DecisionTree models that are executed by each thread of the process.
    *@last_thread is the number of DecisionTrees left for the last thread after symmetric distribution of the trees  
    */
    EXPORT STREAMED DATASET(reg_prediction) RAND_REGTREE(DATASET(NumericField)
    independents, DATASET(NumericField) dependents,INTEGER nthreads=1,INTEGER ntrees=1,
    UNSIGNED maxDepth=100):=EMBED
    (PYTHON : globalscope('globalscope'),persist('query'),activity)
        import pandas as pd
        from sklearn import tree
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.tree import export_text
        import threading
        import math
        global reg_dectrees
        reg_dectrees=[]
        threads=[]
        independents=list(independents)
        dependents=list(dependents)
        outRecs = []
        fields = []
        prevId = 0
        for item in independents:
            wi, id, number, value = item

            if id != prevId :
                if prevId > 0:
                    outRecs.append(fields)
                    fields = []

                prevId = id
            fields.append(value)
        X=pd.DataFrame(outRecs)

        outRecs = []
        fields = []
        prevId = 0
        for item in dependents:
            wi, id, number, value = item

            if id != prevId :
                if prevId > 0:
                    outRecs.append(fields)
                    fields = []

                prevId = id
            fields.append(value)
        Y=pd.DataFrame(outRecs)
        
        

        last_thread=ntrees%nthreads
        ntrees_perthread=math.floor(ntrees/nthreads)
        if(last_thread!=0):
            ntrees_perthread+=1
            last_thread=ntrees-(ntrees_perthread*(nthreads-1))
        def Dec_tree_reg(nt):
            dec_tree_reg=RandomForestRegressor(n_estimators=nt,max_depth=maxDepth)
            dec_tree_reg.fit(X,Y)
            reg_dectrees.append(dec_tree_reg)
            
        for i in range(nthreads):
            if((nthreads-i)==1):
                t=threading.Thread(target=Dec_tree_reg,args=(last_thread,))
                t.start()
                threads.append(t)
            else:
                t=threading.Thread(target=Dec_tree_reg,args=(ntrees_perthread,))
                t.start()
                threads.append(t)

        for t in range(nthreads):
            threads[t].join()
   
        return []
   
    ENDEMBED;


    /* CLASS_PREDICT is use to obtain the prediction for classification data from the  decision trees obtained through DECTREE_CLASS by applying a testing dataset on them.The output is a 2D list of predictions.
    @trees is the set of decision trees obtained from DECTREE_CLASS
    @testing_data is the numeric field dataset applied for prediction
    @class_predict is the output list of predictions in a 2D list format that contains the prediction output
    */      

    EXPORT STREAMED DATASET(DiscreteField) CLASS_PREDICT(DATASET(class_prediction)datas,DATASET(NumericField)testing_data):=FUNCTION
        STREAMED DATASET(class_prediction) PREDICT(DATASET(class_prediction)trees
        ,DATASET(NumericField)testing_data)
        :=EMBED
        (PYTHON : globalscope('globalscope'),persist('query'),activity)
            global class_predict
            class_predict=[]
            import pandas as pd
            outRecs = []
            fields = []
            prevId = 0
            for item in testing_data:
                wi, id, number, value = item

                if id != prevId :
                    if prevId > 0:
                        outRecs.append(fields)
                        fields = []

                    prevId = id
                fields.append(value)
            test=pd.DataFrame(outRecs)
            
            for i in range(len(class_dectrees)):
                dectree_class= class_dectrees[i]
                predict=list(dectree_class.predict(test))
                prediction=[int(x) for x in predict]
                class_predict.append(prediction)
            return class_predict
        ENDEMBED;
        prediction:=PREDICT(datas,testing_data);
        MyDSWithId := PROJECT(prediction, TRANSFORM(class_prediction_id, SELF.id := COUNTER, SELF := LEFT));
 
        NFds := NORMALIZE(MyDSWithId, COUNT(LEFT.prediction), TRANSFORM(DiscreteField, 

            SELF.wi := 1,  

            SELF.id := LEFT.id, 

            SELF.number := COUNTER, 

            SELF.value := LEFT.prediction[COUNTER]));  
        return NFds;
    END;
    /* REG_PREDICT is use to obtain the prediction for regression data from the  decision trees obtained through DECTREE_REG by applying a testing dataset on them.The output is a 2D list of predictions.
    @trees is the set of decision trees obtained from DECTREE_REG
    @testing_data is the numeric field dataset applied for prediction
    @reg_predict is the output list of predictions in a 2D list format that contains the prediction output
    */      
    EXPORT STREAMED DATASET(NumericField)REG_PREDICT(DATASET(reg_prediction)datas,DATASET(NumericField)testing_data):=FUNCTION

        STREAMED DATASET(reg_prediction) PREDICT(DATASET(reg_prediction)datas,DATASET(NumericField)testing_data):=
        EMBED
        (PYTHON : globalscope('globalscope'),persist('query'),activity)
            import pandas as pd
            global reg_predict
            reg_predict=[]
            outRecs = []
            fields = []
            prevId = 0
            for item in testing_data:
                wi, id, number, value = item

                if id != prevId :
                    if prevId > 0:
                        outRecs.append(fields)
                        fields = []

                    prevId = id
                fields.append(value)
            test=pd.DataFrame(outRecs)
            for i in range(len(reg_dectrees)):
                dec_tree_reg=reg_dectrees[i]
                p=list(dec_tree_reg.predict(test))
                reg_predict.append(p)
            return reg_predict
        ENDEMBED;
        prediction:=PREDICT(datas,testing_data);
        MyDSWithId := PROJECT(prediction, TRANSFORM(reg_prediction_id, SELF.id := COUNTER, SELF := LEFT));
 
        NFds := NORMALIZE(MyDSWithId, COUNT(LEFT.prediction), TRANSFORM(NumericField,  

            SELF.wi := 1,  

            SELF.id := LEFT.id, 

            SELF.number := COUNTER,  

            SELF.value := LEFT.prediction[COUNTER]));  
        return NFds;
    END;
    /*CLASS_RESULT is used to obtain the classification result obtained through multiple tree predictions by taking the mode of ṭḥē prediction list
    @class_prediction is the list of predictions obtained from various trees
    @res contains an 1D list of final predictions
    */
    EXPORT STREAMED DATASET(DiscreteField) CLASS_RESULT(STREAMED DATASET(DiscreteField)prediction_list):=FUNCTION
        STREAMED DATASET(class_prediction) RESULT(STREAMED DATASET(DiscreteField)prediction_list):=EMBED
        (PYTHON : globalscope('globalscope'),persist('query'),activity)
            import statistics
            from statistics import mode
            res=[]
            lis=list(class_predict)
            for i in range(len(lis[0])):
                r=[lis[j][i] for j in range(len(lis))]
                r1=mode(r)
                res.append(r1)
            return [res]
    
        ENDEMBED;
        c_result:=RESULT(prediction_list);
        MyDSWithId := PROJECT(c_result, TRANSFORM(class_prediction_id, SELF.id := COUNTER, SELF := LEFT));
 
        NFds := NORMALIZE(MyDSWithId, COUNT(LEFT.prediction), TRANSFORM(DiscreteField,  

            SELF.wi := 1, 

            SELF.id := LEFT.id, 

            SELF.number := COUNTER,  

            SELF.value := LEFT.prediction[COUNTER])); 
        return NFds;
     
    END;
    /*REG_RESULT is used to obtain the regression result obtained through multiple tree predictions by taking the mean of ṭḥē prediction list
    @creg_prediction is the list of predictions obtained from various trees
    @res contains an 1D list of final predictions
    */
    EXPORT STREAMED DATASET(NumericField)REG_RESULT(STREAMED DATASET(NumericField) prediction_list):=FUNCTION
        
        STREAMED DATASET(reg_prediction) RRESULT(STREAMED DATASET(NumericField)
        prediction_list):=EMBED
        (PYTHON : globalscope('globalscope'),persist('query'),activity)
                import statistics
                from statistics import mean
                lis=list(reg_dectrees)
                res=[]
                for i in range(len(lis[0])):
                    r=[lis[j][i] for j in range(len(lis))]
                    r1=mean(r)
                    res.append(r1)
                return [res]
        ENDEMBED;   
        r_result:=RRESULT(prediction_list);
        MyDSWithId := PROJECT(r_result, TRANSFORM(reg_prediction_id, SELF.id := COUNTER, SELF := LEFT));
 
        NFds := NORMALIZE(MyDSWithId, COUNT(LEFT.prediction), TRANSFORM(NumericField, 

            SELF.wi := 1, 

            SELF.id := LEFT.id,

            SELF.number := COUNTER, 

            SELF.value := LEFT.prediction[COUNTER]));  
        return NFds;
    END;
    /* CLASS_MODEL is the function that returns the classification tree model that was used for prediction as a list of stings
    @class_prediction is the input list containing the set of classification predictions
    @class_model returns the final classification tree model as list of string*/
    EXPORT STREAMED DATASET(tree_model) CLASS_MODEL(STREAMED DATASET(class_prediction)
    prediction_list):=EMBED
    (PYTHON : globalscope('globalscope'),persist('query'))
        from sklearn.tree import export_text
        class_model=[]
        for t in class_dectrees:
          for m in t.estimators_:
            class_model.append(export_text(m))
        return class_model;
    ENDEMBED;
    /* REG_MODEL is the function that returns the regression tree model that was used for prediction as a list of stings
    @reg_prediction is the input list containing the set of regression predictions
    @reg_model returns the final reegression tree model as list of string*/
    EXPORT STREAMED DATASET(tree_model) REG_MODEL(STREAMED DATASET(reg_prediction)
    prediction_list):=EMBED
    (PYTHON : globalscope('globalscope'),persist('query'))
        from sklearn.tree import export_text
        reg_model=[]
        for t in reg_dectrees:
          for m in t.estimators_:
            reg_model.append(export_text(m))
        return reg_model;
       
    ENDEMBED;

    
END;
