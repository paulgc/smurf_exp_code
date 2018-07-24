
import copy
import networkx as nx
import itertools

class Node:
    def __init__(self, node_type, predicate, parent=None):
        self.node_type = node_type
        self.predicate = predicate
        self.parent = parent
        self.children = []
        self.predicates = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def remove_child(self, child_node):
        for i in range(len(self.children)):
            if self.children[i].predicate == child_node.predicate:
                break
        self.children.pop(i)

class Plan:
    def __init__(self):
        self.root = Node('ROOT', None)

    def _print(self):                                                           
        print ('ROOT')                                                          
        print ('--------')                                                      
        print ('level 1')                                                       
        for node in self.root.children:                                         
            print (node.predicate, 'root')                                 
        print ('--------')                                                      
        lev_1_ch = self.root.children                                           
        print ('level 2')                                                       
        for node in lev_1_ch:                                                   
            for n in node.children:                                             
                if n.node_type == 'OUTPUT':                                     
                    name = 'output'                                             
                else:                                                           
                    name = n.predicate                                     
                print (name, n.parent.predicate)                           
        print ('---------') 

    def show(self):
        g = nx.DiGraph()
        g.add_node('root')
        for child_node in self.root.children:
            g.add_node(child_node.predicate)
            g.add_edge('root', child_node.predicate)
        curr_list = self.root.children
        while len(curr_list) > 0:
            curr_node = curr_list[0]
            curr_list.pop(0)
            for child_node in curr_node.children:
                g.add_node(child_node.predicate)
                g.add_edge(curr_node.predicate, child_node.predicate)
            curr_list.extend(curr_node.children)
        import matplotlib.pyplot as plt
        nx.draw(g, with_labels=True)   
        plt.savefig('labels.png')
        plt.close()

def get_predicate_dict(rule_sets):
    predicate_dict = {}
    for rule_set in rule_sets:
        for rule in rule_set.rules:
            for predicate in rule.predicates:
                predicate_dict[predicate.name] = predicate
    return predicate_dict

def merge_plans(plan1, plan2, predicate_dict):
    existing_join_nodes = plan1.root.children
    plan2_node = plan2.root.children[0]

    if plan2_node.node_type != 'JOIN':
        print 'INVALID PLAN'
        return

    plan2_node_pred = predicate_dict[plan2_node.predicate]
    plan1_node_to_be_merged = None
    for join_node in existing_join_nodes:
        plan1_node_pred = predicate_dict[join_node.predicate]
        if plan1_node_pred.feat_name != plan2_node_pred.feat_name:
            continue
        plan1_node_to_be_merged = join_node

    if plan1_node_to_be_merged is None:
        plan1.root.add_child(plan2.root.children[0])                            
        return
     
    plan1_node_pred = predicate_dict[plan1_node_to_be_merged.predicate]

    if ((plan1_node_pred.threshold < plan2_node_pred.threshold) or                  
        (plan1_node_pred.threshold == plan2_node_pred.threshold and                 
         plan1_node_pred.comp_op == '>=' and plan2_node_pred.comp_op == '>')):      
        plan2_node.node_type = 'SELECT'                         
        plan2_node.parent = plan1_node_to_be_merged                        
        plan1_node_to_be_merged.add_child(plan2_node)                      
                                                                                
    elif ((plan1_node_pred.threshold > plan2_node_pred.threshold) or                
          (plan1_node_pred.threshold == plan2_node_pred.threshold and               
           plan1_node_pred.comp_op == '>' and plan2_node_pred.comp_op == '>=')):    
        plan1_node_to_be_merged.node_type = 'SELECT'                       
        plan2_node.add_child(plan1_node_to_be_merged)                      
        plan2_node.parent = plan1.root                 
        plan1.root.add_child(plan2_node)               
        plan1.root.remove_child(plan1_node_to_be_merged)          
        plan1_node_to_be_merged.parent = plan2_node                        
                                                                                
    elif plan1_node_pred.threshold == plan2_node_pred.threshold:                    
        plan2_node.children[0].parent = plan1_node_to_be_merged            
        plan2_node = plan2_node.children[0]                     
        plan1_node_to_be_merged.add_child(plan2_node.children[0])   

def merge_filter_nodes(plan):
    output_nodes = []

    queue = []
    for node in plan.root.children:
        queue.append(node)

    while len(queue) > 0:
        node = queue.pop(0)
        if node.node_type == 'OUTPUT':
            output_nodes.append(node)
            continue
        for child_node in node.children:
            queue.append(child_node)

    print 'num output nodes : ', len(output_nodes)
    
    for node in output_nodes:

        nodes_to_merge = []        
        curr = node.parent
        while curr.node_type == 'FILTER':
            nodes_to_merge.append(curr)
            curr = curr.parent

        new_node = Node('FILTER', None, curr)
        for i in xrange(len(nodes_to_merge)-1, -1, -1):
            new_node.predicates.append(nodes_to_merge[i].predicate)
        new_node.add_child(node)

        curr.remove_child(nodes_to_merge[-1])
        curr.add_child(new_node)
        node.parent = new_node         

def merge_plans1(plan1, plan2, predicate_dict):
    sibling_nodes_in_plan1 = plan1.root.children
    plan2_node = plan2.root.children[0]
    no_common_join_predicate = True                                         
    while True:
        continue_merge = False
        pred2 = predicate_dict[plan2_node.predicate]
        for sibling_node in sibling_nodes_in_plan1:
            print 'sib : ', sibling_node.node_type
            pred1 = predicate_dict[sibling_node.predicate]
            if nodes_can_be_merged(sibling_node, plan2_node, pred1, pred2):
                print 't1', plan2_node.node_type
                if plan2_node.node_type == 'JOIN':

                    if ((pred1.threshold < pred2.threshold) or 
                        (pred1.threshold == pred2.threshold and 
                         pred1.comp_op == '>=' and pred2.comp_op == '>')):
                        print 't2'
                        plan2_node.node_type = 'SELECT'
                        plan2_node.parent = sibling_node
                        sibling_node.add_child(plan2_node)
                        no_common_join_predicate = False

                    elif ((pred1.threshold > pred2.threshold) or
                          (pred1.threshold == pred2.threshold and    
                           pred1.comp_op == '>' and pred2.comp_op == '>=')):
                        print 't3'                           
                        sibling_node.node_type = 'SELECT'
                        plan2_node.add_child(sibling_node)
                        plan2_node.parent = sibling_node.parent
                        sibling_node.parent.add_child(plan2_node)
                        sibling_node.parent.remove_child(sibling_node)
                        sibling_node.parent = plan2_node
                        no_common_join_predicate = False                           

                    elif pred1.threshold == pred2.threshold:
                        print 't4'
                        plan2_node.children[0].parent = sibling_node
                        sibling_nodes_to_check = sibling_node.children
                        plan2_node = plan2_node.children[0]                                           
                        sibling_node.add_child(plan2_node.children[0])
                        continue_merge = True
                        no_common_join_predicate = False                                
 
#                elif (plan2_node.node_type == 'SELECT' and 
#                      pred1.threshold == pred2.threshold):
#                    plan2_node.parent.remove_child(plan2_node)                
#                    plan2_node.children[0].parent = sibling_node  
#                    sibling_nodes_to_check = sibling_node.children              
#                    plan2_node = plan2_node.children[0]     
#                    sibling_node.add_child(plan2_node.children[0])
#                    continue_merge = True

                elif plan2_node.node_type == 'FILTER':
                    plan2_node.node_type = 'SELECT'
                    sibling_node.node_type = 'SELECT'
                    new_node = Node('FEATURE', sibling_node.predicate, 
                                    sibling_node.parent)
                    parent_node = sibling_node.parent
                    parent_node.remove_child(sibling_node)
                    new_node.add_child(sibling_node)
                    new_node.add_child(plan2_node)
                    sibling_node.parent = new_node
                    plan2_node.parent = new_node
                    parent_node.add_child(new_node)

                #no_common_join_predicate = False
                break

        if no_common_join_predicate:
            break
        if not continue_merge:
            break

    if no_common_join_predicate:
        plan1.root.add_child(plan2.root.children[0])                 


def nodes_can_be_merged(node1, node2, pred1, pred2):
    if node1.node_type != node2.node_type:
        return False
    if pred1.feat_name != pred2.feat_name:
        return False
    return are_comp_ops_compatible(pred1.comp_op, pred2.comp_op, node1.node_type)  

def are_comp_ops_compatible(comp_op1, comp_op2, node_type):
    if node_type == 'FILTER':
        return True
    if node_type == 'SELECT':
        return comp_op1 == comp_op2
    if comp_op1 in ['<', '<='] and comp_op2 in ['>' '>=']:
        return False
    if comp_op1 in ['>', '>='] and comp_op2 in ['<', '<=']:
        return False
    return True        

def generate_execution_plan(rule_sets):
    plans = []
    for rule_set in rule_sets:
        for rule in rule_set.rules:
            plans.append(get_optimal_plan_for_rule(rule))

    predicate_dict = get_predicate_dict(rule_sets)                              
    curr_plan = copy.deepcopy(plans[0])                                         
    for i in range(1, len(plans)):                                              
        merge_plans(curr_plan, copy.deepcopy(plans[i]), predicate_dict)         
    return curr_plan             

def get_ind_opt_plans(rule_sets):
    plans = []                                                                  
    for rule_set in rule_sets:                                                  
        for rule in rule_set.rules:                                             
            plans.append(get_optimal_plan_for_rule(rule))     
    return plans

def get_ind_opt_plans1(rule_sets):                                               
    plans = []                                                                  
    for rule_set in rule_sets:                                                  
        for rule in rule_set.rules:

            valid_predicates = []                                                       
            invalid_predicates = []                                                     
            for i in xrange(len(rule.predicates)):                                                
                if rule.predicates[i].is_valid_join_predicate():                                 
                    valid_predicates.append(i)                                                                    
                else:                                                                   
                    invalid_predicates.append(i)

            optimal_predicate_seq = []
            
            sample_size = len(rule.predicates[0].coverage)                                                  
            selected_predicates = {}                                                    
            max_score = 0                                                               
            max_pred = -1                                                         
            prev_coverage = None                                                        
            for i in valid_predicates:                                      
                pred_score = (1.0 - (sum(rule.predicates[i].coverage) /                
                    sample_size)) / rule.predicates[i].cost     
                                                                                
                if pred_score > max_score:                                              
                    max_score = pred_score                                              
                    max_pred = i                                                  
                                                                                
            optimal_predicate_seq.append(max_pred)              
            selected_predicates[max_pred] = True                                  
            prev_coverage = rule.predicates[max_pred].coverage                   
                                                                                
            while len(optimal_predicate_seq) != len(valid_predicates):                  
                max_score = -1                                                          
                max_pred = -1                                                     
                                                                                
                for i in valid_predicates:                                  
                    if selected_predicates.get(i) != None:                              
                        continue                                                        
                                                                                
                    combined_coverage = rule.predicates[i].coverage & prev_coverage    
                    pred_score = (1.0 - (sum(combined_coverage) /                       
                        sample_size)) / rule.predicates[i].cost             
                    print pred_score, max_score, sum(prev_coverage), sum(combined_coverage)
                    if pred_score > max_score:                                          
                        max_score = pred_score                                          
                        max_pred = i                                              
                                                                                
                optimal_predicate_seq.append(max_pred)          
                selected_predicates[max_pred] = True                              
                prev_coverage = prev_coverage & rule.predicates[max_pred].coverage
                                                                                
            optimal_predicate_seq.extend(invalid_predicates)    
                                                    
            plan = Plan()                                                               
            curr_node = plan.root                                                       
            join_pred = True                                                            
            for i in optimal_predicate_seq:                                     
                if join_pred:                                                           
                    new_node = Node('JOIN', rule.predicates[i].name, curr_node)                  
                    curr_node.add_child(new_node)                                       
                    curr_node = new_node                                                
                    join_pred = False                                                   
                else:                                                                   
                    new_node = Node('FILTER', rule.predicates[i].name, curr_node)                
                    curr_node.add_child(new_node)                                       
                    curr_node = new_node                                                
            curr_node.add_child(Node('OUTPUT', 'out_'+rule.name, curr_node))
            plans.append(plan)           
    return plans

def generate_execution_plan1(plans, rule_sets):                                         
    predicate_dict = get_predicate_dict(rule_sets)                              
    curr_plan = copy.deepcopy(plans[0])                                         
    for i in range(1, len(plans)):                                              
        merge_plans(curr_plan, copy.deepcopy(plans[i]), predicate_dict)         
    return curr_plan       

def get_optimal_plan_for_rule(rule):
    optimal_predicate_seq = get_optimal_predicate_seq(rule.predicates)

    plan = Plan()                                                               
    curr_node = plan.root                                                       
    join_pred = True        
    for predicate in optimal_predicate_seq:    
        if join_pred:
            new_node = Node('JOIN', predicate.name, curr_node)
            curr_node.add_child(new_node)
            curr_node = new_node
            join_pred = False
        else:
            new_node = Node('FILTER', predicate.name, curr_node)
            curr_node.add_child(new_node)
            curr_node = new_node
    curr_node.add_child(Node('OUTPUT', 'out_'+rule.name, curr_node))
    return plan

def get_optimal_predicate_seq(predicates):
    valid_predicates = []                                                       
    invalid_predicates = []                                                     
    for predicate in predicates:                                           
        if predicate.is_valid_join_predicate():                                 
            valid_predicates.append(predicate)
            print predicate.feat_name                                  
        else:                                                                   
            invalid_predicates.append(predicate)  
    if len(valid_predicates) == 0:
        print 'invalid rf'

    optimal_predicate_seq = []                                                  
    selected_predicates = {}                                                    
    max_score = 0                                                               
    max_pred_index = -1                                                         
    prev_coverage = None                                                        
    for i in range(len(valid_predicates)):                                      
        pred_score = (1.0 - (sum(valid_predicates[i].coverage) /                
             len(valid_predicates[i].coverage))) / valid_predicates[i].cost      
                                                                                
        if pred_score > max_score:                                              
            max_score = pred_score                                              
            max_pred_index = i                                                  
                                                                                
    optimal_predicate_seq.append(valid_predicates[max_pred_index])              
    selected_predicates[max_pred_index] = True                                  
    prev_coverage = valid_predicates[max_pred_index].coverage                   
                                                                                
    while len(optimal_predicate_seq) != len(valid_predicates):                  
        max_score = -1                                                          
        max_pred_index = -1                                                     
                                                                                
        for i in range(len(valid_predicates)):                                  
            if selected_predicates.get(i) != None:                              
                continue                                                        
                                                                                
            combined_coverage = valid_predicates[i].coverage & prev_coverage         
            pred_score = (1.0 - (sum(combined_coverage) /                       
                len(combined_coverage))) / valid_predicates[i].cost             
            print pred_score, max_score, sum(prev_coverage), sum(combined_coverage)                                                                    
            if pred_score > max_score:                                          
                max_score = pred_score                                          
                max_pred_index = i                                              
                                                                                
        optimal_predicate_seq.append(valid_predicates[max_pred_index])          
        selected_predicates[max_pred_index] = True                              
        prev_coverage = prev_coverage & valid_predicates[max_pred_index].coverage

    optimal_predicate_seq.extend(invalid_predicates)
    return optimal_predicate_seq

def select_optimal_set_of_trees(rule_sets):
    num_trees = len(rule_sets)
    min_trees_to_apply = (num_trees / 2) + 1 
    min_score = 1000000
    min_subset_indices = None
    for comb in itertools.combinations(range(len(rule_sets)), min_trees_to_apply):
        score = compute_score_for_trees(map(lambda i: rule_sets[i], comb))            
        if score < min_score:
            min_score = score
            min_subset_indices = comb
    trees_to_apply_over_join = []
    trees_to_apply_over_candset = []
    for i in range(len(rule_sets)):
        if i in comb:
            trees_to_apply_over_join.append(rule_sets[i])
        else:
            trees_to_apply_over_candset.append(rule_sets[i])
    return (trees_to_apply_over_join, trees_to_apply_over_candset)

def foo(rule_sets, plans):
    num_trees = len(rule_sets)
    min_trees_to_apply = (num_trees / 2) + 1                                    
    start = 0
    ind_plans_per_tree = {}

    for i in xrange(num_trees):
        num_rules = len(rule_sets[i].rules)
        ind_plans_per_tree[i] = plans[start:start+num_rules]
        start = start + num_rules

    predicate_dict = get_predicate_dict(rule_sets)                              
    for comb in itertools.combinations(range(num_trees), min_trees_to_apply):
        print comb
        ind_plans = []
        for i in comb:
            ind_plans.extend(ind_plans_per_tree[i])
        pl = generate_execution_plan1(ind_plans, rule_sets)
        print comb, compute_plan_cost(pl.root, None, predicate_dict) 



def compute_score_for_trees(rule_sets):
    return

def generate_greedy_execution_plan(rule_sets):
    naive_plan = generate_execution_plan(rule_sets)
    naive_plan_cost = compute_plan_cost(naive_plan.root, None)
    greedy_plan = Plan()
    max_reduction_pred = -1
    max_reduced_cost = naive_plan_cost
    predicate_dict = {}
    rule_dict = {}
    for rule_set in rule_sets:
        for rule in rule_set.rules:
            rule_dict[rule.name] = rule
            for predicate in rule.predicates:
                if predicate.is_valid_join_predicate():                
                    if pred_dict.get(predicate.feat_name) is None:
                        predicate_dict[predicate.feat_name] = []
                    predicate_dict[predicate.feat_name].append((rule_set.name,
                                                                rule.name,
                                                                predicate.name))
                    break
    
    for feat_name in predicate_dict.keys():
        rule_sets_copy = copy.deepcopy(rule_sets)
        new_rule_set
        for entry in predicate_dict.get(feat_name):
            new_rule_set = RuleSet()
                   
        

def compute_plan_cost(plan_node, coverage, predicate_dict):
    if  plan_node.node_type == 'OUTPUT':
        return 0

    if plan_node.node_type == 'ROOT':
       cost = 0
       for child_node in plan_node.children:
           cost += compute_plan_cost(child_node, coverage, predicate_dict)
       return cost       

    if plan_node.node_type == 'SELECT':
        return 0

    curr_coverage = predicate_dict[plan_node.predicate].coverage
    if plan_node.parent.node_type != 'ROOT':
        curr_coverage = curr_coverage & coverage

    child_nodes_cost = 0
    for child_node in plan_node.children:
        child_nodes_cost += compute_plan_cost(child_node, curr_coverage, predicate_dict)

    if plan_node.parent.node_type == 'ROOT':
        return predicate_dict[plan_node.predicate].cost + child_nodes_cost
    else:
        sel = sum(coverage) / len(coverage)
        return sel * predicate_dict[plan_node.predicate].cost + child_nodes_cost

def recursive_merge(plan_node, index):
    if plan_node.node_type == 'ROOT':
        for child_index in range(len(plan_node.children)):
            recursive_merge(plan_node.children, )
