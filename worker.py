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
        freqItemSet , rules, support_dict =  utils.algo_apiori.findApiroriWithRules(
            dataPath=self.dataPath,
            minSupport=self.minSupport,
            minConfidence=self.minConfidence,
            minRatingFilter=3.5)
        return freqItemSet, rules, support_dict
    
    def runApioriAlgorithm(self):
        freqItemSet =  utils.algo_apiori.findApriori(
            dataPath=self.dataPath,
            minSupport=self.minSupport,
            minConfidence=self.minConfidence,
            minRatingFilter=3.5)
        for k, itemsets in freqItemSet.items():
            print(f"List of {k}-itemsets: {itemsets}")
        return freqItemSet

    def runFPGrowthAlgorithm(self):
        _, transactions = utils.utils.getItemSetAndTransactionList(
            dataPath=self.dataPath, 
            minRating=3.5)
        result =  utils.algo_fpgrowth.fp_growth(transactions,min_support_percentage =  self.minSupport)
        dictionary = {}
        for itemset, support in result.items():
            length = len(itemset)
            if length not in dictionary:
                dictionary[length] = []
            
            dictionary[length].append({
                'items': itemset,
                'support': support
            })
        return dictionary

def run_with_timer(func, name):
    start_time = time.time()
    func()
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"{name} execution time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    worker = Worker(dataPath="movies_dataset/cropped.csv", minSupport=0.1, minConfidence=0.5)
    # # thread1 = threading.Thread(target=run_with_timer, args=(worker.runApioriAlgorithmWithRules, "Apriori"))
    # thread2 = threading.Thread(target=run_with_timer, args=(worker.runFPGrowthAlgorithm, "FP-Growth"))
    # # thread1.start()
    # thread2.start()
    # # thread1.join()
    # thread2.join()

    worker.runFPGrowthAlgorithm()
    