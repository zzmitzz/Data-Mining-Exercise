import itertools
from collections import defaultdict


class HashTreeNode:
    def __init__(self, depth=0, is_leaf=True, max_leaf_size=3):
        self.is_leaf = is_leaf
        self.children = dict()
        self.itemsets = []
        self.counts = dict()
        self.depth = depth
        self.max_leaf_size = max_leaf_size

    def insert(self, itemset):
        itemset = tuple(sorted(itemset))
        if self.is_leaf:
            self.itemsets.append(itemset)
            self.counts[itemset] = 0
            if len(self.itemsets) > self.max_leaf_size:
                self.split()
        else:
            # Kiểm tra nếu depth không vượt quá độ dài itemset
            if self.depth < len(itemset):
                key = self.hash_func(itemset[self.depth])
                if key not in self.children:
                    self.children[key] = HashTreeNode(depth=self.depth+1, max_leaf_size=self.max_leaf_size)
                self.children[key].insert(itemset)

    def split(self):
        self.is_leaf = False
        old_itemsets = self.itemsets[:]  # Tạo bản copy
        self.itemsets = []
        self.counts = {}
        
        for itemset in old_itemsets:
            if self.depth < len(itemset):
                key = self.hash_func(itemset[self.depth])
                if key not in self.children:
                    self.children[key] = HashTreeNode(depth=self.depth+1, max_leaf_size=self.max_leaf_size)
                self.children[key].insert(itemset)

    def hash_func(self, item):
        return hash(item) % 5

    def count_support(self, transaction, k):
        """Đếm support cho tất cả subset có kích thước k trong transaction"""
        subsets = itertools.combinations(sorted(transaction), k)
        for subset in subsets:
            self._update_count(tuple(subset))

    def _update_count(self, itemset):
        """Cập nhật count cho itemset trong hash tree"""
        self._traverse_and_count(itemset, 0)
    
    def _traverse_and_count(self, itemset, depth):
        if self.is_leaf:
            if itemset in self.counts:
                self.counts[itemset] += 1
        else:
            if depth < len(itemset):
                key = self.hash_func(itemset[depth])
                if key in self.children:
                    self.children[key]._traverse_and_count(itemset, depth + 1)

    def get_frequent_itemsets(self, minsup, num_transactions):
        result = {}
        if self.is_leaf:
            for itemset, count in self.counts.items():
                support = count / num_transactions
                if support >= minsup:
                    result[itemset] = support
        else:
            for child in self.children.values():
                result.update(child.get_frequent_itemsets(minsup, num_transactions))
        return result


def generate_candidates_apriori(prev_frequent_itemsets, k):
    """Tạo candidates theo thuật toán Apriori chuẩn (join step + prune step)"""
    candidates = []
    prev_list = list(prev_frequent_itemsets)
    
    # Join step
    for i in range(len(prev_list)):
        for j in range(i + 1, len(prev_list)):
            itemset1 = prev_list[i]
            itemset2 = prev_list[j]
            
            # Kiểm tra điều kiện join: k-2 phần tử đầu giống nhau
            if itemset1[:-1] == itemset2[:-1]:
                # Tạo candidate mới
                candidate = tuple(sorted(set(itemset1) | set(itemset2)))
                if len(candidate) == k:
                    # Prune step: kiểm tra tất cả subset (k-1) có trong prev_frequent_itemsets không
                    subsets = list(itertools.combinations(candidate, k-1))
                    if all(tuple(sorted(subset)) in prev_frequent_itemsets for subset in subsets):
                        candidates.append(candidate)
    
    return list(set(candidates))  # Loại bỏ duplicate


def apriori_hash_tree(transactions, minsup=0.5, minconf=0.6):
    num_transactions = len(transactions)
    all_frequent_itemsets = dict()
    
    print(f"Tổng số transaction: {num_transactions}")
    print(f"Min support: {minsup}, Min confidence: {minconf}")
    print()
    
    # Step 1: Find 1-itemsets
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[(item,)] += 1
    
    L1 = {}
    for itemset, count in item_counts.items():
        support = count / num_transactions
        if support >= minsup:
            L1[itemset] = support
    
    print(f"L1 (1-itemsets thường xuyên): {len(L1)} itemsets")
    for itemset, support in sorted(L1.items()):
        print(f"  {itemset}: support = {support:.3f}")
    print()
    
    all_frequent_itemsets.update(L1)

    k = 2
    prev_L = L1
    
    while prev_L:
        print(f"=== Tìm L{k} ===")
        
        # Step 2: Generate candidate itemsets
        candidates = generate_candidates_apriori(prev_L, k)
        print(f"Số candidates C{k}: {len(candidates)}")
        
        if not candidates:
            break

        # Step 3: Build hash tree và insert candidates
        tree = HashTreeNode()
        for c in candidates:
            tree.insert(c)

        # Step 4: Count support
        for transaction in transactions:
            tree.count_support(transaction, k)

        # Step 5: Extract frequent itemsets
        Lk = tree.get_frequent_itemsets(minsup, num_transactions)
        
        print(f"L{k} (frequent {k}-itemsets): {len(Lk)} itemsets")
        for itemset, support in sorted(Lk.items()):
            print(f"  {itemset}: support = {support:.3f}")
        print()
        
        all_frequent_itemsets.update(Lk)
        prev_L = Lk
        k += 1

    # Step 6: Generate association rules
    print("=== Tạo Association Rules ===")
    rules = []
    
    for itemset in all_frequent_itemsets:
        if len(itemset) < 2:
            continue
            
        # Tạo tất cả subset không rỗng của itemset (trừ chính nó)
        for r in range(1, len(itemset)):
            for A in itertools.combinations(itemset, r):
                A = tuple(sorted(A))
                B = tuple(sorted(set(itemset) - set(A)))
                
                if len(B) == 0:
                    continue
                    
                support_AB = all_frequent_itemsets[itemset]
                support_A = all_frequent_itemsets.get(A, 0)
                
                if support_A > 0:
                    confidence = support_AB / support_A
                    if confidence >= minconf:
                        rules.append({
                            'antecedent': set(A),
                            'consequent': set(B),
                            'rule': f"{set(A)} => {set(B)}",
                            'support': round(support_AB, 4),
                            'confidence': round(confidence, 4)
                        })

    # Sắp xếp rules theo confidence giảm dần
    rules.sort(key=lambda x: x['confidence'], reverse=True)
    
    print(f"Tổng số association rules tìm được: {len(rules)}")
    for rule in rules:
        print(f"  {rule['rule']}: support = {rule['support']}, confidence = {rule['confidence']}")

    return all_frequent_itemsets, rules


# Test với dữ liệu mẫu
if __name__ == "__main__":
    # Dữ liệu từ bảng của bạn
    transactions = [
        ['milk', 'bread', 'butter'],
        ['bread', 'butter'],
        ['milk', 'bread'],
        ['milk', 'bread', 'butter'],
        ['bread']      # T500
    ]
    
    # Loại bỏ duplicate items trong mỗi transaction
    transactions = [list(set(t)) for t in transactions]
    
    print("Dữ liệu transactions:")
    for i, t in enumerate(transactions, 1):
        print(f"T{i}00: {sorted(t)}")
    print()
    
    # Chạy Apriori với min_sup = 60%, min_conf = 80%
    frequent_itemsets, rules = apriori_hash_tree(transactions, minsup=0.6, minconf=0.8)