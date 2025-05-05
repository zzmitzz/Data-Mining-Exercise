import itertools
import pandas as pd
from itertools import chain, combinations
import utils.utils
from collections import defaultdict

def genNextKItemset(prevFreqItemSet, k):
    """
    Generate candidate (k)-itemsets by joining frequent (k-1)-itemsets.
    """
    nextItemSet = set()
    prevFreqItemSet = list(prevFreqItemSet)
    for i in range(len(prevFreqItemSet)):
        for j in range(i + 1, len(prevFreqItemSet)):
            # Join step: find itemsets that share k-2 items
            item1 = prevFreqItemSet[i]
            item2 = prevFreqItemSet[j]
            
            # Check if first k-2 elements are the same to enable efficient joining
            if k > 2 and item1[:-1] != item2[:-1]:
                continue
                
            # Create new candidate itemset
            union = set(item1).union(set(item2))
            if len(union) == k:
                candidate = tuple(sorted(union))
                nextItemSet.add(candidate)
    
    return nextItemSet

def pruneCandidate(candidate_itemsets, prev_frequent_itemsets, k):
    """
    Prune candidate k-itemsets if any of their (k-1)-subsets are not frequent.
    """
    pruned_candidates = set()
    prev_frequent_set = set(prev_frequent_itemsets)
    
    for candidate in candidate_itemsets:
        should_prune = False
        # Generate all (k-1)-subsets of the candidate
        for subset in combinations(candidate, k-1):
            # If any subset is not frequent, prune the candidate
            if subset not in prev_frequent_set:
                should_prune = True
                break
                
        if not should_prune:
            pruned_candidates.add(candidate)
            
    return pruned_candidates
"""
    Count support for each itemset and return those that meet minimum support threshold.
"""
def getItemsSetAndMinSupport(itemSets, transactions, minSupport):
    
    _innerDictSet = defaultdict(int)
    transaction_sets = [set(t) for t in transactions]

    for item in itemSets:
        item_set = set(item)
        for transaction in transaction_sets:
            if item_set.issubset(transaction):
                _innerDictSet[item] += 1

    resultItemSet = set()
    support_dict = {}
    for item in itemSets:
        support_count = _innerDictSet[item]
        supportValue = support_count / len(transactions)
        if supportValue >= minSupport:
            resultItemSet.add(item)
            support_dict[item] = supportValue

    return resultItemSet, support_dict




def findFrequentItemSet(initItemSet, transactions, minSupport):
    """
    Find all frequent itemsets using the Apriori algorithm.
    """
    frequentKItemSetDict = dict()
    support_dict = {}
    
    # First pass to find frequent 1-itemsets
    L_k_set, k1_support = getItemsSetAndMinSupport(initItemSet, transactions, minSupport)
    support_dict.update(k1_support)
    
    currentSet = L_k_set  # Start with frequent 1-itemsets
    lengthK = 2
    
    while currentSet:
        frequentKItemSetDict[lengthK-1] = currentSet
        
        # Generate candidates for next level
        candidate_set = genNextKItemset(currentSet, lengthK)
        
        pruned_candidates = pruneCandidate(candidate_set, currentSet, lengthK)
        # Find frequent k-itemsets
        currentSet, k_support = getItemsSetAndMinSupport(pruned_candidates, transactions, minSupport)
        support_dict.update(k_support)
        
        lengthK += 1

    return frequentKItemSetDict, support_dict

def generateAssociationRules(frequentItemSets, support_dict, minConfidence):
    """
    Generate association rules from frequent itemsets with support and confidence.
    """
    rules = []
    for k, itemsets in frequentItemSets.items():
        if k < 2:
            continue  # need at least 2 items for rules
        for itemset in itemsets:
            itemset_support = support_dict[itemset]
            itemset_set = set(itemset)
            
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    antecedent = tuple(sorted(antecedent))
                    consequent_set = itemset_set - set(antecedent)
                    consequent = tuple(sorted(consequent_set))
                    
                    antecedent_support = support_dict[antecedent]
                    confidence = itemset_support / antecedent_support
                    
                    if confidence >= minConfidence:
                        rules.append({
                            "antecedent": antecedent,
                            "consequent": consequent,
                            "support": itemset_support,
                            "confidence": confidence,
                        })
    return rules

def findApriori(
        dataPath="ml-latest/cropped.csv", 
        minSupport = 0.2, 
        minConfidence = 0.5,
        minRatingFilter = 3.5):
    """
    Run the complete Apriori algorithm 
    """
    itemSet, transactions = utils.utils.getItemSetAndTransactionList(
        dataPath=dataPath,
        minRating=minRatingFilter)
    
    freqItemSets, support_dict = findFrequentItemSet(
        initItemSet=itemSet,
        transactions=transactions,
        minSupport=minSupport
    )
    
    return freqItemSets

def findApiroriWithRules(
        dataPath="ml-latest/cropped.csv", 
        minSupport = 0.2, 
        minConfidence = 0.5,
        minRatingFilter = 3.5):
    """
    Run the complete Apriori algorithm and return frequent itemsets and rules.
    """
    itemSet, transactions = utils.utils.getItemSetAndTransactionList(
        dataPath=dataPath,
        minRating=minRatingFilter)
    
    freqItemSets, support_dict = findFrequentItemSet(
        initItemSet=itemSet,
        transactions=transactions,
        minSupport=minSupport
    )

    rules = generateAssociationRules(
        frequentItemSets=freqItemSets,
        support_dict=support_dict,
        minConfidence=minConfidence
    )
    
    return freqItemSets, rules
