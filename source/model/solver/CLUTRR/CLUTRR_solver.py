import pickle

def solve(completion):
    ''' Execute the completion to solve the question (for CLUTRR).
    :param completion (str): the model completion

    :return (str): the final relation
    '''

    # get the relation from every step
    relations = []
    for line in completion.split('\n')[1:]:
        if ' = ' in line and not '@' in line: # it's a single relation line, not a comment line or the final relation deduction line
            try:
                relation = line.split(' = ')[1] # get the relation
            except IndexError as e:
                print(f"Error: {e}, line: {line}")
                continue
            relations.append(relation)

    # load the transitive rules
    with open("source/model/solver/CLUTRR/trans_rules.pkl", 'rb') as f:
        trans_dict = pickle.load(f)
      
     # apply the transitive rules to get the final relation
    if not relations:
        return "[invalid]"

    final_relation = ""
    for relation in reversed(relations):
        if not final_relation:
            # first relation
            final_relation = relation
        else:
            # apply transitive rules
            try:
                final_relation = trans_dict[(final_relation, relation)]
            except KeyError:
                return "[invalid]"
    return final_relation

if __name__ == "__main__":
    # run a simple test
    import os
    os.chdir("../../../..")
    blob = '''
# 1. How is [Lisa] related to [Michelle]? (independent, support: "[Michelle] is married to Thomas and when she was 24, the couple welcomed [Lisa] into the world.")
relation(Lisa, Michelle) = daughter
# 2. How is [Michelle] related to [Molly]? (independent, support: "[Michelle] was excited for today, its her daughter's, [Molly], spring break. She will finally get to see her.")
relation(Michelle, Molly) = mother
# 3. How is [Molly] related to [Valerie]? (independent, support: "[Molly] took her daughter [Valerie] to school during the cold winter morning.")
relation(Molly, Valerie) = mother
# 4. Final answer: How is [Lisa] related to [Valerie]? (depends on 1, 2, 3)
relation(Lisa, Valerie) = relation(Lisa, Michelle) @ relation(Michelle, Molly) @ relation(Molly, Valerie)
'''
    print(solve(blob))