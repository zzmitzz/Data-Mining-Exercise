import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import time
from scipy import sparse
import warnings
import utils.algo_apiori
import utils.algo_fpgrowth
import utils.utils
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

# Read the ratings data
def load_data(file_path):
    return pd.read_csv(file_path)

# Convert ratings to transactions with memory optimization
def create_transactions(df, rating_threshold=3.5):
    # Filter ratings above threshold
    df_filtered = df[df['rating'] >= rating_threshold]
    
    # Group by userId and create lists of movieIds
    transactions = df_filtered.groupby('userId')['movieId'].apply(list).values.tolist()
    return transactions

def run_library_apriori(transactions, min_support=0.1):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    sparse_matrix = sparse.csr_matrix(te_ary)
    df_encoded = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=te.columns_)
    df_encoded.columns = [str(i) for i in df_encoded.columns]
    
    start_time = time.time()
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    end_time = time.time()
    
    result_dict = {}
    for _, row in frequent_itemsets.iterrows():
        items = tuple(row['itemsets'])
        length = len(items)
        if length not in result_dict:
            result_dict[length] = []
        result_dict[length].append({
            'items': items,
            'support': row['support']
        })
    
    return result_dict, end_time - start_time

def run_library_fpgrowth(transactions, min_support=0.1):
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    sparse_matrix = sparse.csr_matrix(te_ary)
    df_encoded = pd.DataFrame.sparse.from_spmatrix(sparse_matrix, columns=te.columns_)
    df_encoded.columns = [str(i) for i in df_encoded.columns]
    
    start_time = time.time()
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    end_time = time.time()
    
    result_dict = {}
    for _, row in frequent_itemsets.iterrows():
        items = tuple(row['itemsets'])
        length = len(items)
        if length not in result_dict:
            result_dict[length] = []
        result_dict[length].append({
            'items': items,
            'support': row['support']
        })
    
    return result_dict, end_time - start_time

def run_manual_apriori(file_path, min_support=0.1, min_confidence=0.5):
    start_time = time.time()
    freqItemSet, _, support_dict = utils.algo_apiori.findApiroriWithRules(
        dataPath=file_path,
        minSupport=min_support,
        minConfidence=min_confidence,
        minRatingFilter=3.5)
    end_time = time.time()
    
    # Transform freq_itemsets to include support values
    result_dict = {}
    for k, itemsets in freqItemSet.items():
        result_dict[k] = [
            {
                'items': itemset,
                'support': support_dict[itemset]
            }
            for itemset in itemsets
        ]
    
    return result_dict, end_time - start_time

def run_manual_fpgrowth(file_path, min_support=0.1):
    start_time = time.time()
    _, transactions = utils.utils.getItemSetAndTransactionList(
        dataPath=file_path, 
        minRating=3.5)
    result = utils.algo_fpgrowth.fp_growth(transactions, min_support_percentage=min_support)
    
    dictionary = {}
    for itemset, support in result.items():
        length = len(itemset)
        if length not in dictionary:
            dictionary[length] = []
        dictionary[length].append({
            'items': tuple(sorted(itemset)) if isinstance(itemset, set) else itemset,
            'support': support
        })
    end_time = time.time()
    return dictionary, end_time - start_time

def print_top_itemsets(itemsets_dict, algorithm_name):
    print(f"\nTop 10 frequent itemsets from {algorithm_name}:")
    print("-" * 50)
    
    for length in sorted(itemsets_dict.keys()):
        itemsets = itemsets_dict[length]
        # Sort by support
        itemsets.sort(key=lambda x: x['support'], reverse=True)
        
        print(f"\n{length}-itemsets:")
        for itemset in itemsets[:10]:
            print(f"Items: {itemset['items']}, Support: {itemset['support']:.4f}")

def calculate_similarity(lib_dict, manual_dict):
    # Convert itemsets to sets for easier comparison
    lib_itemsets = set()
    manual_itemsets = set()
    
    for length in lib_dict:
        for itemset in lib_dict[length]:
            # Handle both tuple and single item cases
            items = itemset['items']
            if isinstance(items, (int, str)):
                items = (items,)
            # Convert string IDs to integers
            items = tuple(int(x) if isinstance(x, str) else x for x in items)
            lib_itemsets.add(frozenset(items))
    
    for length in manual_dict:
        for itemset in manual_dict[length]:
            # Handle both tuple and single item cases
            items = itemset['items']
            if isinstance(items, (int, str)):
                items = (items,)
            # Convert string IDs to integers
            items = tuple(int(x) if isinstance(x, str) else x for x in items)
            manual_itemsets.add(frozenset(items))
    
    # Calculate Jaccard similarity
    intersection = len(lib_itemsets.intersection(manual_itemsets))
    union = len(lib_itemsets.union(manual_itemsets))
    
    if union == 0:
        return 0.0
    
    return (intersection / union) * 100

def compare_results(lib_dict, manual_dict, algorithm_name):
    print(f"\nComparing {algorithm_name} results:")
    print("-" * 50)
    
    # Compare number of itemsets
    lib_total = sum(len(itemsets) for itemsets in lib_dict.values())
    manual_total = sum(len(itemsets) for itemsets in manual_dict.values())
    print(f"Library found {lib_total} itemsets")
    print(f"Manual implementation found {manual_total} itemsets")
    
    # Calculate similarity
    similarity = calculate_similarity(lib_dict, manual_dict)
    print(f"\nSimilarity between implementations: {similarity:.2f}%")
    
    # Compare itemsets by length
    for length in sorted(set(lib_dict.keys()) | set(manual_dict.keys())):
        lib_itemsets = lib_dict.get(length, [])
        manual_itemsets = manual_dict.get(length, [])
        
        # Convert to sets of itemsets for comparison
        lib_itemset_set = {frozenset(int(x) if isinstance(x, str) else x for x in itemset['items']) 
                          for itemset in lib_itemsets}
        manual_itemset_set = {frozenset(int(x) if isinstance(x, str) else x for x in itemset['items']) 
                            for itemset in manual_itemsets}
        
        # Find common and unique itemsets
        common_itemsets = lib_itemset_set.intersection(manual_itemset_set)
        lib_unique = lib_itemset_set - manual_itemset_set
        manual_unique = manual_itemset_set - lib_itemset_set
        
        print(f"\n{length}-itemsets:")
        print(f"Total itemsets - Library: {len(lib_itemsets)}, Manual: {len(manual_itemsets)}")
        print(f"Common itemsets: {len(common_itemsets)}")
        print(f"Itemsets only in Library: {len(lib_unique)}")
        print(f"Itemsets only in Manual: {len(manual_unique)}")
    

def plot_comparison(support_values, results):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Number of Itemsets Comparison
    x = np.arange(len(support_values))
    width = 0.2
    
    # Plot bars for each method
    ax1.bar(x - 1.5*width, results['Library Apriori Itemsets'], width, 
            label='Library Apriori', color='skyblue')
    ax1.bar(x - 0.5*width, results['Manual Apriori Itemsets'], width, 
            label='Manual Apriori', color='lightgreen')
    ax1.bar(x + 0.5*width, results['Library FPGrowth Itemsets'], width, 
            label='Library FPGrowth', color='salmon')
    ax1.bar(x + 1.5*width, results['Manual FPGrowth Itemsets'], width, 
            label='Manual FPGrowth', color='lightcoral')
    
    ax1.set_xlabel('Minimum Support')
    ax1.set_ylabel('Number of Itemsets')
    ax1.set_title('Number of Itemsets Found')
    ax1.set_xticks(x)
    ax1.set_xticklabels(support_values)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Execution Time Comparison
    ax2.bar(x - 1.5*width, results['Library Apriori Time'], width, 
            label='Library Apriori', color='skyblue')
    ax2.bar(x - 0.5*width, results['Manual Apriori Time'], width, 
            label='Manual Apriori', color='lightgreen')
    ax2.bar(x + 0.5*width, results['Library FPGrowth Time'], width, 
            label='Library FPGrowth', color='salmon')
    ax2.bar(x + 1.5*width, results['Manual FPGrowth Time'], width, 
            label='Manual FPGrowth', color='lightcoral')
    
    ax2.set_xlabel('Minimum Support')
    ax2.set_ylabel('Execution Time (seconds)')
    ax2.set_title('Execution Time')
    ax2.set_xticks(x)
    ax2.set_xticklabels(support_values)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('algorithm_comparison.png')
    plt.close()

def plot_support_comparison(support_values, results):
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Number of Itemsets vs Min Support
    plt.subplot(2, 2, 1)
    for label, counts in results.items():
        if 'Itemsets' in label:
            plt.plot(support_values, counts, 'o-', label=label, linewidth=2, markersize=8)
    plt.xlabel('Minimum Support')
    plt.ylabel('Number of Itemsets')
    plt.title('Number of Itemsets vs Minimum Support')
    plt.legend()
    plt.grid(True)
    
    # 2. Execution Time vs Min Support
    plt.subplot(2, 2, 2)
    for label, times in results.items():
        if 'Time' in label:
            plt.plot(support_values, times, 'o-', label=label, linewidth=2, markersize=8)
    plt.xlabel('Minimum Support')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time vs Minimum Support')
    plt.legend()
    plt.grid(True)
    
    # 3. Itemsets by Length for each Min Support
    plt.subplot(2, 2, 3)
    for i, min_support in enumerate(support_values):
        lib_lengths = results[f'Library Lengths {min_support}']
        manual_lengths = results[f'Manual Lengths {min_support}']
        x = np.arange(len(lib_lengths))
        width = 0.35
        
        plt.bar(x - width/2 + i*0.1, lib_lengths, width, 
                label=f'Lib (s={min_support})' if i == 0 else "", alpha=0.7)
        plt.bar(x + width/2 + i*0.1, manual_lengths, width, 
                label=f'Manual (s={min_support})' if i == 0 else "", alpha=0.7)
    
    plt.xlabel('Itemset Length')
    plt.ylabel('Number of Itemsets')
    plt.title('Itemsets by Length for Different Min Support Values')
    plt.legend()
    plt.grid(True)
    
    # 4. Support Distribution for each Min Support
    plt.subplot(2, 2, 4)
    for min_support in support_values:
        lib_supports = results[f'Library Supports {min_support}']
        manual_supports = results[f'Manual Supports {min_support}']
        
        plt.hist([lib_supports, manual_supports], bins=20, 
                label=[f'Lib (s={min_support})', f'Manual (s={min_support})'],
                alpha=0.5)
    
    plt.xlabel('Support Value')
    plt.ylabel('Frequency')
    plt.title('Support Value Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('support_analysis.png')
    plt.close()
    
    # Create additional plots
    # 5. Venn Diagram for each Min Support
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, min_support in enumerate(support_values):
        lib_set = results[f'Library Sets {min_support}']
        manual_set = results[f'Manual Sets {min_support}']
        
        from matplotlib_venn import venn2
        venn2([lib_set, manual_set], 
              set_labels=(f'Library (s={min_support})', f'Manual (s={min_support})'),
              ax=axes[i])
        axes[i].set_title(f'Common Itemsets (min_support={min_support})')
    
    plt.tight_layout()
    plt.savefig('venn_diagrams.png')
    plt.close()
    
    # 6. Support Comparison Scatter Plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, min_support in enumerate(support_values):
        lib_supports = results[f'Library Supports {min_support}']
        manual_supports = results[f'Manual Supports {min_support}']
        
        axes[i].scatter(lib_supports, manual_supports, alpha=0.5)
        axes[i].plot([0, 1], [0, 1], 'r--')  # Diagonal line
        axes[i].set_xlabel('Library Support')
        axes[i].set_ylabel('Manual Support')
        axes[i].set_title(f'Support Comparison (min_support={min_support})')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('support_scatter.png')
    plt.close()

def main():
    try:
        # Load data
        file_path = r"C:\Users\Admin\Documents\Last semester\Data Mining Exercise\movies_dataset\ratings_small.csv"
        print("Loading data...")
        df = load_data(file_path)
        
        # Create transactions with limited number of items
        print("\nCreating transactions...")
        transactions = create_transactions(df, rating_threshold=3.5)
        print(f"Number of transactions: {len(transactions)}")
        
        # Test different minimum support values
        min_supports = [0.05, 0.1, 0.15, 0.2, 0.25]
        min_support = 0.1
        min_confidence = 0.5
        print("Running Library Implementations...")
        print("\nLibrary Apriori:")
        lib_apriori_itemsets, lib_apriori_time = run_library_apriori(transactions, min_support)
        print(f"Execution time: {lib_apriori_time:.2f} seconds")
        print_top_itemsets(lib_apriori_itemsets, "Library Apriori")
        
        print("\nLibrary FPGrowth:")
        lib_fpgrowth_itemsets, lib_fpgrowth_time = run_library_fpgrowth(transactions, min_support)
        print(f"Execution time: {lib_fpgrowth_time:.2f} seconds")
        print_top_itemsets(lib_fpgrowth_itemsets, "Library FPGrowth")
        
        print("\nRunning Manual Implementations...")
        print("\nManual Apriori:")
        manual_apriori_itemsets, manual_apriori_time = run_manual_apriori(file_path, min_support, min_confidence)
        print(f"Execution time: {manual_apriori_time:.2f} seconds")
        print_top_itemsets(manual_apriori_itemsets, "Manual Apriori")
        
        print("\nManual FPGrowth:")
        manual_fpgrowth_itemsets, manual_fpgrowth_time = run_manual_fpgrowth(file_path, min_support)
        print(f"Execution time: {manual_fpgrowth_time:.2f} seconds")
        print_top_itemsets(manual_fpgrowth_itemsets, "Manual FPGrowth")
        
        print("\nPerformance Comparison:")
        print("-" * 50)
        print(f"Library Apriori time: {lib_apriori_time:.2f} seconds")
        print(f"Manual Apriori time: {manual_apriori_time:.2f} seconds")
        print(f"Library FPGrowth time: {lib_fpgrowth_time:.2f} seconds")
        print(f"Manual FPGrowth time: {manual_fpgrowth_time:.2f} seconds")
        
        print("\nResults Comparison:")
        compare_results(lib_apriori_itemsets, manual_apriori_itemsets, "Apriori")
        compare_results(lib_fpgrowth_itemsets, manual_fpgrowth_itemsets, "FPGrowth")
        # Store results for plotting
        results = {
            'Library Apriori Itemsets': [],
            'Manual Apriori Itemsets': [],
            'Library FPGrowth Itemsets': [],
            'Manual FPGrowth Itemsets': [],
            'Library Apriori Time': [],
            'Manual Apriori Time': [],
            'Library FPGrowth Time': [],
            'Manual FPGrowth Time': []
        }
        
        print("\nRunning algorithms with different minimum support values...")
        for min_support in min_supports:
            print(f"\nMinimum Support: {min_support}")
            
            # Run Library Apriori
            lib_apriori_itemsets, lib_apriori_time = run_library_apriori(transactions, min_support)
            total_lib_itemsets = sum(len(itemsets) for itemsets in lib_apriori_itemsets.values())
            results['Library Apriori Itemsets'].append(total_lib_itemsets)
            results['Library Apriori Time'].append(lib_apriori_time)
            
            # Run Manual Apriori
            manual_apriori_itemsets, manual_apriori_time = run_manual_apriori(file_path, min_support, min_confidence)
            total_manual_itemsets = sum(len(itemsets) for itemsets in manual_apriori_itemsets.values())
            results['Manual Apriori Itemsets'].append(total_manual_itemsets)
            results['Manual Apriori Time'].append(manual_apriori_time)
            
            # Run Library FPGrowth
            lib_fpgrowth_itemsets, lib_fpgrowth_time = run_library_fpgrowth(transactions, min_support)
            total_lib_fpgrowth = sum(len(itemsets) for itemsets in lib_fpgrowth_itemsets.values())
            results['Library FPGrowth Itemsets'].append(total_lib_fpgrowth)
            results['Library FPGrowth Time'].append(lib_fpgrowth_time)
            
            # Run Manual FPGrowth
            manual_fpgrowth_itemsets, manual_fpgrowth_time = run_manual_fpgrowth(file_path, min_support)
            total_manual_fpgrowth = sum(len(itemsets) for itemsets in manual_fpgrowth_itemsets.values())
            results['Manual FPGrowth Itemsets'].append(total_manual_fpgrowth)
            results['Manual FPGrowth Time'].append(manual_fpgrowth_time)
            
            print(f"Library Apriori: {total_lib_itemsets} itemsets, {lib_apriori_time:.2f} seconds")
            print(f"Manual Apriori: {total_manual_itemsets} itemsets, {manual_apriori_time:.2f} seconds")
            print(f"Library FPGrowth: {total_lib_fpgrowth} itemsets, {lib_fpgrowth_time:.2f} seconds")
            print(f"Manual FPGrowth: {total_manual_fpgrowth} itemsets, {manual_fpgrowth_time:.2f} seconds")
        
        # Create visualization
        plot_comparison(min_supports, results)
        print("\nComparison plot has been saved as 'algorithm_comparison.png'")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
