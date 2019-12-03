from ObservationTable import ObservationTable
import DFA
from time import clock

def run_lstar(teacher,time_limit):
    table = ObservationTable(teacher.alphabet,teacher)
    start = clock()
    teacher.counterexample_generator.set_time_limit(time_limit,start)
    table.set_time_limit(time_limit,start)

    while True:
        while True:
            while table.find_and_handle_inconsistency():
                pass
            if table.find_and_close_row():
                continue
            else:
                break
        dfa = DFA.DFA(obs_table=table)
        print("obs table refinement took " + str(int(1000*(clock()-start))/1000.0) )
        counterexample = teacher.equivalence_query(dfa)
        if None is counterexample:
            break
        start = clock()
        table.add_counterexample(counterexample,teacher.classify_word(counterexample))
    return dfa