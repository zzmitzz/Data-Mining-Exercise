import networkx as nx
import matplotlib.pyplot as plt

class FPNode:
    """A node in the FP tree."""
    def __init__(self, item, count=1, parent=None):
        self.item = item  # Item represented by this node
        self.count = count  # Count of this item
        self.parent = parent  # Parent node
        self.children = {}  # Dictionary of children nodes
        self.next = None  # Link to the next similar item in the FP-tree
        self.node_id = None  # Unique ID for visualization

    def increment(self, count=1):
        """Increment the count of this node by the given amount."""
        self.count += count

    def display(self, indent=1):
        """Display the node and its children."""
        print(' ' * indent, self.item, ':', self.count)
        for child in self.children.values():
            child.display(indent + 1)


class FPTree:
    """FP Tree for the FP-Growth algorithm."""
    def __init__(self):
        self.root = FPNode(None)  # Root node with null item
        self.header_table = {}  # Header table for each item
        self.min_support = 0  # Minimum support count
        self.node_count = 0  # Counter for assigning unique IDs
        
    def _get_next_id(self):
        """Get a unique ID for a node."""
        self.node_count += 1
        return self.node_count
        
    def _update_header_table(self, item, node):
        """Update the header table by adding a link to the given node."""
        if item in self.header_table:
            current = self.header_table[item]
            # Find the last node in the linked list
            while current.next is not None:
                current = current.next
            current.next = node
        else:
            # Create a new entry in the header table
            self.header_table[item] = node
            
    def _insert_tree(self, transaction, count):
        """Insert a transaction into the tree."""
        current = self.root
        
        for item in transaction:
            if item in current.children:
                # If the item exists, increment its count
                current.children[item].increment(count)
            else:
                # Create a new node
                new_node = FPNode(item, count, current)
                new_node.node_id = self._get_next_id()
                current.children[item] = new_node
                # Update the header table
                self._update_header_table(item, new_node)
                
            current = current.children[item]
    
    def create_fptree_from_transactions(self, transactions, min_support):
        """
        Create an FP-tree from a list of transactions.
        
        Args:
            transactions: List of transactions, where each transaction is a list of items
            min_support: Minimum support threshold (count)
        
        Returns:
            Frequency of each item
        """
        self.min_support = min_support
        
        # Count the frequency of each item
        item_count = {}
        for transaction in transactions:
            for item in transaction:
                item_count[item] = item_count.get(item, 0) + 1
        
        print(f"Item counts: {item_count}")
        
        # Filter out items below minimum support
        frequent_items = {item: count for item, count in item_count.items() 
                         if count >= min_support}
        
        print(f"Frequent items: {frequent_items}")
        
        if len(frequent_items) == 0:
            return None
        
        # Sort frequent items by frequency in each transaction
        for transaction in transactions:
            # Filter items and sort by frequency
            filtered_transaction = [item for item in transaction if item in frequent_items]
            filtered_transaction.sort(key=lambda item: (-frequent_items[item], item))
            
            if filtered_transaction:
                print(f"Inserting: {filtered_transaction}")
                self._insert_tree(filtered_transaction, 1)
        
        return frequent_items
    
    def visualize_tree(self):
        """Visualize the FP-tree using networkx and matplotlib."""
        G = nx.DiGraph()
        
        # Assign ID to root
        self.root.node_id = 0
        
        # Add root node
        G.add_node(self.root.node_id, label=f"Root")
        
        # Queue for BFS traversal
        queue = [(self.root, None)]
        
        while queue:
            node, parent_id = queue.pop(0)
            
            # Process children
            for child in node.children.values():
                # Add node
                G.add_node(child.node_id, label=f"{child.item}:{child.count}")
                
                # Add edge from parent
                G.add_edge(node.node_id, child.node_id)
                
                # Add to queue
                queue.append((child, child.node_id))
        
        # Add header table links (with dashed lines)
        for item, node in self.header_table.items():
            # Add links between nodes with the same item
            current = node
            while current.next is not None:
                G.add_edge(current.node_id, current.next.node_id, style='dashed', color='red')
                current = current.next
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
        
        # Draw solid edges (tree structure)
        solid_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style', 'solid') == 'solid']
        nx.draw_networkx_edges(G, pos, edgelist=solid_edges)
        
        # Draw dashed edges (header links)
        dashed_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('style') == 'dashed']
        nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, style='dashed', edge_color='red')
        
        # Draw labels
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels=labels)
        
        plt.title("FP-Tree Visualization")
        plt.axis('off')
        
        return plt
        
    def _find_prefix_path(self, node):
        """
        Find the prefix path ending at the given node.
        
        Args:
            node: Node in the FP-tree
            
        Returns:
            List of paths, where each path is a list of (item, count) tuples
        """
        prefix_path = []
        
        # Go up from the node to the root, collecting items along the way
        current = node.parent
        while current.item is not None:  # Stop at the root
            prefix_path.append(current.item)
            current = current.parent
            
        return prefix_path, node.count
    
    def _mine_tree(self, header_table, prefix, frequent_patterns, min_support):
        """
        Mine the FP-tree recursively.
        
        Args:
            header_table: Header table for the current FP-tree
            prefix: Current prefix pattern
            frequent_patterns: Dictionary to store frequent patterns
            min_support: Minimum support threshold
        """
        # Sort header table items by frequency (ascending)
        sorted_items = sorted(header_table.keys(), 
                              key=lambda item: self.header_table[item].count)
        
        for item in sorted_items:
            # Generate a new frequent pattern by adding the current item to the prefix
            new_pattern = prefix.copy()
            new_pattern.append(item)
            
            # Add the pattern to the result with its support
            node = header_table[item]
            support = 0
            while node is not None:
                support += node.count
                node = node.next
            
            pattern_str = frozenset(new_pattern)
            frequent_patterns[pattern_str] = support
            
            print(f"\nMining for item: {item} (with prefix {prefix})")
            print(f"Found pattern: {new_pattern} with support {support}")
            
            # Construct conditional pattern base
            conditional_pattern_base = []
            
            node = header_table[item]
            while node is not None:
                prefix_path, count = self._find_prefix_path(node)
                if prefix_path:
                    print(f"  Prefix path: {prefix_path} (count: {count})")
                    conditional_pattern_base.append((prefix_path, count))
                node = node.next
            
            if conditional_pattern_base:
                print(f"  Conditional pattern base for {item}: {conditional_pattern_base}")
            else:
                print(f"  No conditional pattern base for {item}")
                continue
            
            # Construct conditional FP-tree
            cond_tree = FPTree()
            
            # Count items in the conditional pattern base
            item_count = {}
            for path, count in conditional_pattern_base:
                for path_item in path:
                    item_count[path_item] = item_count.get(path_item, 0) + count
            
            print(f"  Item counts in conditional base: {item_count}")
            
            # Filter out items below minimum support
            freq_items = {item: count for item, count in item_count.items() 
                         if count >= min_support}
            
            print(f"  Frequent items in conditional base: {freq_items}")
            
            if freq_items:
                print(f"  Building conditional FP-tree for {item}")
                
                # Insert each prefix path into the conditional FP-tree
                for path, count in conditional_pattern_base:
                    # Filter and sort items
                    filtered_path = [p for p in path if p in freq_items]
                    filtered_path.sort(key=lambda x: (-freq_items[x], x))
                    
                    if filtered_path:
                        print(f"    Inserting path: {filtered_path} (count: {count})")
                        cond_tree._insert_tree(filtered_path, count)
                
                # Create new header table for the conditional FP-tree
                cond_header_table = {}
                for path_item in freq_items:
                    if path_item in cond_tree.header_table:
                        cond_header_table[path_item] = cond_tree.header_table[path_item]
                
                # Recursively mine the conditional FP-tree
                if cond_header_table:
                    print(f"  Recursively mining conditional tree with header: {list(cond_header_table.keys())}")
                    cond_tree._mine_tree(cond_header_table, new_pattern, frequent_patterns, min_support)
                else:
                    print(f"  No header table for conditional tree, stopping recursion")
            else:
                print(f"  No frequent items in conditional pattern base, stopping recursion")


def fp_growth(transactions, min_support_percentage):
    """
    FP-Growth algorithm to find frequent itemsets.
    
    Args:
        transactions: List of transactions, where each transaction is a list of items
        min_support_percentage: Minimum support threshold as a percentage (0-1)
        
    Returns:
        Dictionary of frequent itemsets with their support counts
    """
    # Calculate minimum support count
    num_transactions = len(transactions)
    min_support = int(min_support_percentage * num_transactions)
    
    print(f"Minimum support count: {min_support} ({min_support_percentage * 100}%)")
    
    # Create FP-tree
    fp_tree = FPTree()
    fp_tree.create_fptree_from_transactions(transactions, min_support)
    
    print("\nFP-Tree structure:")
    fp_tree.root.display()
    
    # Visualize the tree
    plt = fp_tree.visualize_tree()
    plt.show()
    
    # Mine the FP-tree to find frequent patterns
    frequent_patterns = {}
    fp_tree._mine_tree(fp_tree.header_table, [], frequent_patterns, min_support)
    
    # Convert frozenset keys to tuple for better readability
    result = {tuple(sorted(pattern)): support 
             for pattern, support in frequent_patterns.items()}
    
    return result


# Example usage with a simple dataset
if __name__ == "__main__":
    # Small sample transactions for easy visualization
    transactions = [
        ['a', 'b', 'c'],
        ['a', 'b', 'd'],
        ['a', 'c', 'd'],
        ['a', 'b', 'c', 'd'],
        ['b', 'c', 'd']
    ]
    
    # Find frequent itemsets with minimum support of 60%
    min_support_percentage = 0.6
    frequent_itemsets = fp_growth(transactions, min_support_percentage)
    
    # Display the results
    print("\nFrequent Itemsets:")
    for itemset, support in sorted(frequent_itemsets.items(), key=lambda x: (-x[1], x[0])):
        print(f"{itemset}: {support}")