
import utils


class Node:
    """A node in the FP tree."""
    def __init__(self, item, count=1, parent=None):
        self.item = item  # Item represented by this node
        self.count = count  # Count of this item
        self.parent = parent  # Parent node
        self.children = {}  # Dictionary of children nodes
        self.next = None  # Link to the next similar item in the FP-tree

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
        self.root = Node(None)  # Root node with null item
        self.header_table = {}  # Header table for each item
        self.min_support = 0  # Minimum support count
        
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
                new_node = Node(item, count, current)
                current.children[item] = new_node
                # Update the header table
                self._update_header_table(item, new_node)
                
            current = current.children[item]
    
    # Step 1: Initial Database Scan
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
        
        # Filter out items below minimum support
        frequent_items = {item: count for item, count in item_count.items() 
                         if count >= min_support}
        
        if len(frequent_items) == 0:
            return None
        
        # Step 2: FP-Tree Construction
        # Sort frequent items by frequency in each transaction
        for transaction in transactions:
            # Filter items and sort by frequency
            filtered_transaction = [item for item in transaction if item in frequent_items]
            filtered_transaction.sort(key=lambda item: (-frequent_items[item], item))
            
            if filtered_transaction:
                self._insert_tree(filtered_transaction, 1)
        
        return frequent_items
    
    def display(self):
        """Display the FP-tree."""
        self.root.display()
        
    def display_header_table(self):
        """Display the header table."""
        print("Header Table:")
        for item, node in self.header_table.items():
            print(f"Item: {item}")
            while node is not None:
                print(f"  Node(count={node.count})")
                node = node.next
    
    # Step 3: Mining Frequent Patterns from FP-Tree
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
            
            # Construct conditional pattern base
            conditional_pattern_base = []
            
            node = header_table[item]
            while node is not None:
                prefix_path, count = self._find_prefix_path(node)
                if prefix_path:
                    conditional_pattern_base.append((prefix_path, count))
                node = node.next
            
            # Construct conditional FP-tree
            cond_tree = FPTree()
            
            # Count items in the conditional pattern base
            item_count = {}
            for path, count in conditional_pattern_base:
                for path_item in path:
                    item_count[path_item] = item_count.get(path_item, 0) + count
            
            # Filter out items below minimum support
            freq_items = {item: count for item, count in item_count.items() 
                         if count >= min_support}
            
            if freq_items:
                # Insert each prefix path into the conditional FP-tree
                for path, count in conditional_pattern_base:
                    # Filter and sort items
                    filtered_path = [p for p in path if p in freq_items]
                    filtered_path.sort(key=lambda x: (-freq_items[x], x))
                    
                    if filtered_path:
                        cond_tree._insert_tree(filtered_path, count)
                
                # Create new header table for the conditional FP-tree
                cond_header_table = {}
                for path_item in freq_items:
                    if path_item in cond_tree.header_table:
                        cond_header_table[path_item] = cond_tree.header_table[path_item]
                
                # Recursively mine the conditional FP-tree
                if cond_header_table:
                    cond_tree._mine_tree(cond_header_table, new_pattern, frequent_patterns, min_support)


# Step 4: Pattern Generation Process (main FP-Growth algorithm)
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
    # Create FP-tree
    fp_tree = FPTree()
    fp_tree.create_fptree_from_transactions(transactions, min_support)
    
    # Mine the FP-tree to find frequent patterns
    frequent_patterns = {}
    fp_tree._mine_tree(fp_tree.header_table, [], frequent_patterns, min_support)
    # Convert frozenset keys to tuple for better readability
    result = {tuple(sorted(pattern)): (support/num_transactions) 
             for pattern, support in frequent_patterns.items()}
    
    return result

