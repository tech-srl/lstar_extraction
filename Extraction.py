from time import clock
from ObservationTable import TableTimedOut
from DFA import DFA
from Teacher import Teacher
from Lstar import run_lstar

def extract(rnn,time_limit = 50,initial_split_depth = 10,starting_examples=None):
	print("provided counterexamples are:",starting_examples)
	guided_teacher = Teacher(rnn,num_dims_initial_split=initial_split_depth,starting_examples=starting_examples)
	start = clock()
	try:
	    run_lstar(guided_teacher,time_limit)
	except KeyboardInterrupt: #you can press the stop button in the notebook to stop the extraction any time
	    print("lstar extraction terminated by user")
	except TableTimedOut:
	    print("observation table timed out during refinement")
	end = clock()
	extraction_time = end-start

	dfa = guided_teacher.dfas[-1]

	print("overall guided extraction time took: " + str(extraction_time))

	print("generated counterexamples were: (format: (counterexample, counterexample generation time))")
	print('\n'.join([str(a) for a in guided_teacher.counterexamples_with_times]))
	return dfa