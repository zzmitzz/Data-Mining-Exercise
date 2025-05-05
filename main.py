from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from utils.algo_apiori import findApiroriWithRules, findFrequentItemSet, generateAssociationRules
import uvicorn
import time
import worker
app = FastAPI()

class CustomDataset(BaseModel):
    data: List[List[str]]
    min_support: Optional[float] = 0.2
    min_confidence: Optional[float] = 0.5
    min_rating_filter: Optional[float] = 3.5

@app.post("/apriori/custom")
async def run_apriori_custom(dataset: CustomDataset):
    start_time = time.time()
    try:
        # Convert the incoming 2D list into the format expected by the algorithm
        itemset = set()
        for transaction in dataset.data:
            for item in transaction:
                itemset.add(tuple([item]))
        
        freq_itemsets, support_dict = findFrequentItemSet(
            initItemSet=itemset,
            transactions=dataset.data,
            minSupport=dataset.min_support
        )
        
        rules = generateAssociationRules(
            frequentItemSets=freq_itemsets,
            support_dict=support_dict,
            minConfidence=dataset.min_confidence
        )
        
        return {
            "frequent_itemsets": freq_itemsets,
            "rules": rules,
            "time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/apriori/default")
async def run_apriori_default(
    min_support: Optional[float] = 0.1,
    min_confidence: Optional[float] = 0.5,
    min_rating_filter: Optional[float] = 3.5
):
    mining = worker.Worker(dataPath="ml-latest/cropped.csv", minSupport=min_support, minConfidence=min_confidence)
    start_time = time.time()
    try:
        freq_itemsets, rules = mining.runApioriAlgorithmWithRules()
        return {
            "frequent_itemsets": freq_itemsets,
            "rules": rules,
            "time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/fpgrowth/default")
async def run_fpgrowth_default(
    min_support_percentage: Optional[float] = 0.1
):
    mining = worker.Worker(dataPath="ml-latest/cropped.csv", minSupport=0.1, minConfidence=0.5)
    start_time = time.time()
    try:
        freq_itemsets = mining.runFPGrowthAlgorithm(min_support_percentage=min_support_percentage)
        return {
            "frequent_itemsets": freq_itemsets,
            "time": time.time() - start_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)