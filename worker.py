
import os
import sys
import utils
import utils.algo_apiori
import utils.algo_fpgrowth
import utils.utils
import pandas as pd
import threading
import time


class Worker:
    def __init__(self, dataPath, minSupport, minConfidence):
        self.dataPath = dataPath
        self.minSupport = minSupport
        self.minConfidence = minConfidence

    def runApioriAlgorithmWithRules(self):
        freqItemSet , rules =  utils.algo_apiori.findApiroriWithRules(
            dataPath=self.dataPath,
            minSupport=self.minSupport,
            minConfidence=self.minConfidence,
            minRatingFilter=3.5)
        for k, itemsets in freqItemSet.items():
            print(f"List of {k}-itemsets: {itemsets}")
        for rule in rules:
            print(f"{rule['antecedent']} -> {rule['consequent']} ; Support: ({rule['support']:.2f}) ; Confidence: ({rule['confidence']:.2f})")
        return freqItemSet, rules
    
    def runApioriAlgorithm(self):
        freqItemSet =  utils.algo_apiori.findApriori(
            dataPath=self.dataPath,
            minSupport=self.minSupport,
            minConfidence=self.minConfidence,
            minRatingFilter=3.5)
        for k, itemsets in freqItemSet.items():
            print(f"List of {k}-itemsets: {itemsets}")
        return freqItemSet
    
    def runFPGrowthAlgorithm(self, min_support_percentage = 0.1):
        _, transactions = utils.utils.getItemSetAndTransactionList(
            dataPath=self.dataPath, 
            minRating=3.5)
        frequent_itemsets = utils.algo_fpgrowth.fp_growth(transactions, min_support_percentage)
        print("Frequent Itemsets:")
        for itemset, support in sorted(frequent_itemsets.items(), key=lambda x: (-x[1], x[0])):
            print(f"{itemset}: {support}")
    

def run_with_timer(func, name):
    start_time = time.time()
    func()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"{name} execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    worker = Worker(dataPath="ml-latest/cropped.csv", minSupport=0.1, minConfidence=0.5)
    thread1 = threading.Thread(target=run_with_timer, args=(worker.runApioriAlgorithmWithRules, "Apriori"))
    thread2 = threading.Thread(target=run_with_timer, args=(worker.runFPGrowthAlgorithm, "FP-Growth"))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()