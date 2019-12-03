from Helper_Functions import n_words_of_length

def make_train_set_for_target(target,alphabet,lengths=None,max_train_samples_per_length=300,search_size_per_length=1000,provided_examples=None):
    train_set = {}
    if None is provided_examples:
        provided_examples = []
    if None is lengths:
        lengths = list(range(15))+[15,20,25,30] 
    for l in lengths:
        samples = [w for w in provided_examples if len(w)==l]
        samples += n_words_of_length(search_size_per_length,l,alphabet)
        pos = [w for w in samples if target(w)]
        neg = [w for w in samples if not target(w)]
        pos = pos[:int(max_train_samples_per_length/2)]
        neg = neg[:int(max_train_samples_per_length/2)]
        minority = min(len(pos),len(neg))
        pos = pos[:minority+20]
        neg = neg[:minority+20]
        train_set.update({w:True for w in pos})
        train_set.update({w:False for w in neg})

    print("made train set of size:",len(train_set),", of which positive examples:",
        len([w for w in train_set if train_set[w]==True]))
    return train_set

#curriculum
def mixed_curriculum_train(rnn,train_set,outer_loops=3,stop_threshold=0.001,learning_rate=0.001,
    length_epochs=5,random_batch_epochs=100,single_batch_epochs=100,random_batch_size=20):
    lengths = sorted(list(set([len(w) for w in train_set])))
    for _ in range(outer_loops):
        for l in lengths:
            training = {w:train_set[w] for w in train_set if len(w)==l}
            if len(set([training[w] for w in training])) <= 1: #empty, or length with only one classification
                continue
            rnn.train_group(training,length_epochs,show=False,loss_every=20,stop_threshold=stop_threshold,
                            learning_rate=learning_rate,batch_size=None,print_time=False)
        # all together but in batches
        if rnn.finish_signal == rnn.train_group(train_set,random_batch_epochs,show=True,loss_every=20,
                                                stop_threshold = stop_threshold,
                                                learning_rate=learning_rate,
                                                batch_size=random_batch_size,print_time=False):
            break
        # all together in one batch
        if rnn.finish_signal == rnn.train_group(train_set,single_batch_epochs,show=True,loss_every=20,
                                                stop_threshold = stop_threshold,
                                                learning_rate=learning_rate,batch_size=None,print_time=False): 
            break
    print("classification loss on last batch was:",rnn.all_losses[-1])