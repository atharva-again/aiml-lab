"""
Graph Traversal: BFS and DFS Implementation
This module demonstrates Breadth-First Search (BFS) and Depth-First Search (DFS)
traversal algorithms on a sample graph with 6 nodes.
"""

import time
import os
import sys
from collections import deque

# Sample Graph Definition
graph = {"A": ["B", "C"], "B": ["D", "E"], "C": ["F"], "D": [], "E": ["F"], "F": []}


def clear_screen():
    """Clears the terminal screen for better UX"""
    os.system("cls" if os.name == "nt" else "clear")


def wait_for_user():
    """Pauses execution until user presses Enter"""
    input("\nPress Enter to continue...")


def bfs(graph, start, delay=1.0):
    """
    Breadth-First Search (BFS) Traversal with Animation

    Uses a queue (FIFO) to explore nodes level by level.
    Time Complexity: O(V + E) where V = vertices, E = edges
    Space Complexity: O(V)

    Args:
        graph (dict): Adjacency list representation of the graph
        start (str): Starting node for traversal
        delay (float): Time delay between steps for animation

    Returns:
        list: Order of nodes visited during BFS traversal
    """
    print(f"\nStarting BFS from node '{start}'...")
    print("Initializing Queue with start node...")
    time.sleep(delay)

    visited = set()
    queue = deque([start])
    traversal_order = []

    visited.add(start)
    print(f"Queue: {list(queue)}")
    print(f"Visited: {visited}")
    time.sleep(delay)

    while queue:
        print("\n" + "-" * 30)
        node = queue.popleft()
        traversal_order.append(node)
        print(f"→ Dequeued node: {node} (Current Node)")
        time.sleep(delay / 2)

        print(f"  Visiting node: {node}")
        time.sleep(delay / 2)

        # Explore all adjacent nodes
        neighbors = graph[node]
        if not neighbors:
            print(f"  Node {node} has no neighbors.")

        for neighbor in neighbors:
            print(f"  Checking neighbor: {neighbor}")
            time.sleep(delay / 2)

            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                print(f"    → {neighbor} is not visited. Added to queue.")
                print(f"    Queue: {list(queue)}")
            else:
                print(f"    → {neighbor} is already visited. Skipping.")
            time.sleep(delay / 2)

    print("\n" + "-" * 30)
    print("Queue is empty. BFS Traversal Complete.")
    return traversal_order


def dfs(graph, start, delay=1.0):
    """
    Depth-First Search (DFS) Traversal using Recursion with Animation

    Explores as far as possible along each branch before backtracking.
    Time Complexity: O(V + E) where V = vertices, E = edges
    Space Complexity: O(V) for recursion stack

    Args:
        graph (dict): Adjacency list representation of the graph
        start (str): Starting node for traversal
        delay (float): Time delay between steps for animation

    Returns:
        list: Order of nodes visited during DFS traversal
    """
    print(f"\nStarting DFS from node '{start}'...")
    time.sleep(delay)

    visited = set()
    traversal_order = []

    def dfs_recursive(node, depth=0):
        """Helper function for recursive DFS"""
        indent = "  " * depth
        print(f"{indent}→ Visiting node: {node}")
        time.sleep(delay)

        visited.add(node)
        traversal_order.append(node)

        neighbors = graph[node]
        if not neighbors:
            print(f"{indent}  (Leaf node, backtracking...)")
            time.sleep(delay / 2)

        # Explore all adjacent nodes
        for neighbor in neighbors:
            print(f"{indent}  Checking neighbor: {neighbor}")
            time.sleep(delay / 2)

            if neighbor not in visited:
                print(f"{indent}    → Moving deeper to {neighbor}")
                dfs_recursive(neighbor, depth + 1)
            else:
                print(f"{indent}    → {neighbor} already visited. Skipping.")
                time.sleep(delay / 2)

        if depth > 0:
            print(f"{indent}  Finished with {node}, returning to previous level.")
            time.sleep(delay / 2)

    dfs_recursive(start)
    print("\nDFS Traversal Complete.")
    return traversal_order


def print_graph_structure():
    """
    Print the visual representation of the graph structure
    Shows the adjacency list and a text-based diagram
    """
    print("\n" + "=" * 60)
    print("GRAPH STRUCTURE VISUALIZATION")
    print("=" * 60)

    print("\nAdjacency List Representation:")
    print("-" * 40)
    for node, neighbors in graph.items():
        print(f"{node} → {neighbors}")

    print("\n\nText-Based Graph Diagram:")
    print("-" * 40)
    print("""
                    A (Root)
                   / \\
                  /   \\
                 B     C
                / \\     \\
               /   \\     \\
              D     E     F
              
    Nodes: A, B, C, D, E, F (Total: 6 nodes)
    Edges: A→B, A→C, B→D, B→E, C→F, E→F
    """)

    print("\nDetailed Node Information:")
    print("-" * 40)
    for node in sorted(graph.keys()):
        neighbors = graph[node]
        if neighbors:
            print(f"Node {node}: Connected to {', '.join(neighbors)}")
        else:
            print(f"Node {node}: Leaf node (no outgoing edges)")


def get_valid_start_node(graph):
    """Prompts user for a valid start node"""
    while True:
        start_node = (
            input(f"\nEnter start node ({', '.join(sorted(graph.keys()))}): ")
            .strip()
            .upper()
        )
        if start_node in graph:
            return start_node
        print(f"Invalid node! Please choose from: {', '.join(sorted(graph.keys()))}")


def main():
    """Main function with interactive menu"""
    current_start_node = "A"

    while True:
        clear_screen()
        print("\n" + "=" * 60)
        print("GRAPH TRAVERSAL INTERACTIVE DEMO")
        print("=" * 60)
        print(f"Current Start Node: {current_start_node}")
        print("-" * 40)
        print("1. Run BFS Traversal")
        print("2. Run DFS Traversal")
        print("3. View Graph Structure")
        print("4. Change Starting Node")
        print("5. Exit")
        print("-" * 40)

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            clear_screen()
            print("\n" + "=" * 60)
            print("BREADTH-FIRST SEARCH (BFS) TRAVERSAL")
            print("=" * 60)
            print(f"\nStarting from node '{current_start_node}':")
            print(
                "Strategy: Explore all neighbors at current depth before going deeper"
            )
            print("-" * 40)
            bfs_result = bfs(graph, current_start_node)
            print(f"\nBFS Traversal Order: {' → '.join(bfs_result)}")
            wait_for_user()

        elif choice == "2":
            clear_screen()
            print("\n" + "=" * 60)
            print("DEPTH-FIRST SEARCH (DFS) TRAVERSAL")
            print("=" * 60)
            print(f"\nStarting from node '{current_start_node}':")
            print(
                "Strategy: Explore as far as possible along each branch before backtracking"
            )
            print("-" * 40)
            dfs_result = dfs(graph, current_start_node)
            print(f"\nDFS Traversal Order: {' → '.join(dfs_result)}")
            wait_for_user()

        elif choice == "3":
            clear_screen()
            print_graph_structure()
            wait_for_user()

        elif choice == "4":
            print("\nAvailable Nodes:")
            print_graph_structure()
            current_start_node = get_valid_start_node(graph)
            print(f"\nStart node updated to '{current_start_node}'")
            time.sleep(1)

        elif choice == "5":
            print("\nExiting... Goodbye!")
            break

        else:
            print("\nInvalid choice! Please try again.")
            time.sleep(1)


if __name__ == "__main__":
    main()
