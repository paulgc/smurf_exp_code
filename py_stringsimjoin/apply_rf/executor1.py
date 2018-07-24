
def execute_plan(plan, ltable, rtable, l_key_attr, r_key_attr,                            
                 l_match_attr, r_match_attr, feature_table, n_jobs=1):

    queue = []
    queue.extend(plan.root.children)
    while len(queue) > 0:
        curr_node = queue.pop(0)
        execute_node(curr_node, ltable, rtable, l_key_attr, r_key_attr,
                     l_match_attr, r_match_attr, feature_table, n_jobs)
        if curr_node.node_type == 'OUTPUT':
            continue
        queue.extend(curr_node.children)

def execute_node(node, ltable, rtable, l_key_attr, r_key_attr,                            
                 l_match_attr, r_match_attr, feature_table, n_jobs):
    if node.node_type == 'JOIN':

    elif node.node_type == 'FILTER':

    elif node.node_type == 'SELECT':

    elif node.node_type == 'OUTPUT':

