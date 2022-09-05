import collections
import operator
import statistics
import numpy as np
class eval():
    def __init__(self, num_classes, pred_label_list, actual_label_list, gen_label_list):
#         super().__init__(pred_label_list, actual_label_list, gen_label_list)
        self.preds = pred_label_list
        self.labels = actual_label_list
        self.gender = gen_label_list
        self.num_classes = num_classes

        self.class_list = list(range(self.num_classes))
        self.class_list
        job_0 = []
        job_1 = []
        job_2 = []
        job_3 = []
        job_4 = []
        job_5 = []
        job_6 = []
        job_7 = []
        job_8 = []
        job_9 = []
        job_10 = []
        job_11 = []
        job_12 = []
        job_13 = []

        self.class_dict = {0:job_0, 1:job_1, 2:job_2, 3:job_3, 4:job_4, 5:job_5, 6:job_6, 7:job_7, 8:job_8, 9:job_9, 10:job_10, 11:job_11, 12:job_12, 13:job_13}
#         class_dict

        for i in self.class_list:
            for ind, label in enumerate(self.labels):
                if label == i:
        #             print(i,preds[ind])
                    self.class_dict[i].append((self.labels[ind], self.preds[ind], self.gender[ind]))
            
            
    # computing true positive rate difference
    def TPR_nums(self, key):
        res_dict = self.class_dict[key]

        F_Y = []
        M_Y = []
        F_Y_predY = []
        M_Y_predY = []
        for i in res_dict:
            if i[2] == 0:
                F_Y.append(i)
                if i[1] == key:
                     F_Y_predY.append(i)
            else:
                M_Y.append(i)
                if i[1] == key:
                    M_Y_predY.append(i)


        return F_Y, M_Y, F_Y_predY, M_Y_predY


    def TPR_diff(self, key):
        F_Y, M_Y, F_Y_predY, M_Y_predY = self.TPR_nums(key)
        alpha = 1E-40
        TPR_d = ((len(F_Y_predY)/(len(F_Y)+alpha)) - (len(M_Y_predY)/(len(M_Y)+alpha)))
        # TPR_b = ((len(F_Y_predY)/(len(F_Y)+alpha))/(len(M_Y_predY)/(len(M_Y)+alpha)))
        return TPR_d


    # Computing false positive rate difference
    def FPR_nums(self, key):
        res_dict = self.class_dict[key]

        F_NotY = []
        M_NotY = []
        F_NotY_predY = []
        M_NotY_predY = []
        
        class_dict = self.class_dict


        for cl in class_dict:
            if cl!=key:
                for i in class_dict[cl]:
                    if i[2]==0:
                        F_NotY.append(i)
                        if i[1]==key:
                            F_NotY_predY.append(i)
                    else:
                        M_NotY.append(i)
                        if i[1]==key:
                            M_NotY_predY.append(i)
        return F_NotY, M_NotY, F_NotY_predY, M_NotY_predY




    def FPR_diff(self, key):
        F_NotY, M_NotY, F_NotY_predY, M_NotY_predY = self.FPR_nums(key)
        alpha = 1E-40
        FPR_d = ((len(F_NotY_predY)/(len(F_NotY)+alpha)) - (len(M_NotY_predY)/(len(M_NotY)+alpha)))
        # FPR_b = ((len(F_NotY_predY)/(len(F_NotY)+alpha))/(len(M_NotY_predY)/(len(M_NotY)+alpha)))
        return FPR_d


    # computing true negative values
    def TNR_nums(self, key):
        res_dict = self.class_dict[key]

        F_NotY = []
        M_NotY = []
        F_NotY_NotpredY = []
        M_NotY_NotpredY = []


        for cl in self.class_dict:
            if cl!=key:
                for i in self.class_dict[cl]:
                    if i[2]==0:
                        F_NotY.append(i)
                        if i[1]!=key:
                            F_NotY_NotpredY.append(i)
                    else:
                        M_NotY.append(i)
                        if i[1]!=key:
                            M_NotY_NotpredY.append(i)
        return F_NotY, M_NotY, F_NotY_NotpredY, M_NotY_NotpredY


    # computing accuracy
    def accuracy(self, key):
        F_Y, M_Y, F_Y_predY, M_Y_predY = self.TPR_nums(key)
        F_NotY, M_NotY, F_NotY_NotpredY, M_NotY_NotpredY = self.TNR_nums(key)
        F_accuracy = (len(F_Y_predY) + len(F_NotY_NotpredY))/(len(F_Y) + len(F_NotY))
        M_accuracy = (len(M_Y_predY) + len(M_NotY_NotpredY))/(len(M_Y) + len(M_NotY))
        return F_accuracy, M_accuracy
    
        # computing accuracy
    def F1(self, key):
        alpha = 1E-40
        F_Y, M_Y, F_Y_predY, M_Y_predY = self.TPR_nums(key)
        F_NotY, M_NotY, F_NotY_NotpredY, M_NotY_NotpredY = self.TNR_nums(key)
        F_NotY, M_NotY, F_NotY_predY, M_NotY_predY = self.FPR_nums(key)
        F_prec = len(F_Y_predY)/(len(F_Y_predY) + len(F_NotY_predY)+alpha)
        F_recall = len(F_Y_predY)/(len(F_Y)+alpha)
        F_F1 = 2*((F_prec*F_recall)/(F_prec+F_recall+alpha))
        M_prec = len(M_Y_predY)/(len(M_Y_predY) + len(M_NotY_predY)+alpha)
        M_recall = len(M_Y_predY) /(len(M_Y)+alpha)
        M_F1 = 2*((M_prec*M_recall)/(M_prec+M_recall+alpha))           
        return F_F1, M_F1


    # Computing counterfactual fairness
    def C_fairness(self, y_test_prob, y_CF_test_prob):
        ll = list(map(operator.sub, y_test_prob, y_CF_test_prob))
        ll = [abs(l) for l in ll]
        return statistics.mean(ll), min(ll), max(ll), np.std(ll)
