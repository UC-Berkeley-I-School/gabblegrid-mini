import networkx as nx
import matplotlib.pyplot as plt

def create_interaction_diagram():
    G = nx.DiGraph()
    G.add_edges_from([
        ("user_proxy", "chat_manager"),
        ("chat_manager", "log_data_retriever"),
        ("log_data_retriever", "chat_manager"),
        ("chat_manager", "user_proxy"),
        ("chat_manager", "log_data_writer"),
        ("log_data_writer", "chat_manager"),
    ])

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=3000, edge_color='black', linewidths=1, font_size=15, arrowsize=20)
    image_path = '/home/ubuntu/efs-w210-capstone-ebs/00.GabbleGrid/project/files/images/interaction_diagram.png'
    plt.savefig(image_path)
    return image_path