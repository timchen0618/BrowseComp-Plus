# BrowseComp-Plus

## Generate QAMPARI Results
Setting: (say we’re evaluating MRecall @ k, e.g. k=100)
1. Retrieve k’ documents every turn (e.g. k’=5)
2. Run until it stops / N search budget (e.g. N=10)  
  a. Combine all the documents returned to be candidate set D’.   
  b. If |D’| >= k: return top-k documents from |D’|, ranked by the turn number  
  c. If |D’| < k: Retrieve (k-|D’|) documents using the last query issued to the retriever, and combine them with |D’|.   

Steps
1. Run `qampari.SBATCH` to get the trajectories
2. Run `generate_qampari_results.py` to parse the last queries. 
3. 