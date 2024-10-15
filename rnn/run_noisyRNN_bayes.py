import glob
import pickle
import numpy as np
import torch, torch.nn as nn, torch.optim as optim

device = torch.device('cpu') 
# shorthand random Generator
rng = np.random.default_rng()

# this function could be put in the higher-level code to not recomputed every time this is run
def sample_dist(nlevel, llrmax): 
    level_list = np.arange(-nlevel, nlevel+1)/nlevel
    llr_list = level_list*llrmax
    p_gen = 1/(1+np.exp(-llr_list))
    p_gen = p_gen/p_gen.sum()
    return p_gen

# block stimulus value generator
def generate_blocks(n_trials, n_episodes, n_blocks):
    # configuration
    epiavg = 12  # avg number of trials per episode
    epimin = round(epiavg/2)   # min number of trials per episode
    epimax = round(epiavg*2)  # max number of trials per episode
    fnr    = 0.3  # false negative rate
    llrmax = 1.7954
    nlevel = 49  # number of stimuli in favor of an option
    
    # calculate probability values
    llr_dist = sample_dist(nlevel, llrmax)

    redo = True
    while redo:
        # generate episode lengths from poisson random variable
        epi_lengths = []
        ctr = 0

        # get episode lengths that add up to n_trials
        while True:
            arr_sample = rng.poisson(epiavg, [n_blocks, n_episodes])
            # take blocks for which sum is 72
            ind_filt = arr_sample.sum(axis=1) == 72
            arr_sample = arr_sample[ind_filt, :]
            # append to list for keeping
            [epi_lengths.append(epi) for epi in arr_sample]
            # count successful generations
            ctr += ind_filt.sum()
            if ctr >= n_blocks:
                break
        
        # trim episode length array to be exact
        epi_lengths = np.array(epi_lengths[0:n_blocks])
        
        i_loop_limit = int(1e4)
        # make sure values are within extrema
        for i in range(i_loop_limit):
            ind_min = epi_lengths < epimin # find loc. less than min
            ind_max = epi_lengths > epimax # find loc. greater than max
            # find row and col indices for correction
            idx_min = np.transpose((ind_min).nonzero())
            idx_max = np.transpose((ind_max).nonzero())
            # 1/ account for smaller than min
            idx_min_row = idx_min[:,0]
            idx_min_col = idx_min[:,1]
            epi_lengths[idx_min_row,idx_min_col] += 1
            # remove one from higher values (if sum unmatched)
            if np.all(epi_lengths[idx_min_row,:] != n_trials):
                idx_high = epi_lengths[idx_min_row,:].argmax(axis=1)
                epi_lengths[idx_min_row,idx_high] -= 1
            
            # 2/ account for greater than max
            idx_max_row = idx_max[:,0]
            idx_max_col = idx_max[:,1]
            epi_lengths[idx_max_row,idx_max_col] -= 1
            # add one from lower values (if sum unmatched)
            if np.all(epi_lengths[idx_max_row,:] != n_trials):
                idx_low = epi_lengths[idx_max_row,:].argmin(axis=1)
                epi_lengths[idx_max_row,idx_low] += 1
        
            # 3/ ensure number of trials
            idx_toohigh_row = epi_lengths.sum(1) > n_trials
            idx_toolow_row  = epi_lengths.sum(1) < n_trials
            while idx_toohigh_row.sum()>0 or idx_toolow_row.sum()>0:
                if idx_toohigh_row.sum()>0:
                    idx_high = epi_lengths[idx_toohigh_row,:].argmax(axis=1)
                    epi_lengths[idx_toohigh_row,idx_high] -= 1
                if idx_toolow_row.sum()>0:
                    idx_low = epi_lengths[idx_toolow_row,:].argmin(axis=1)
                    epi_lengths[idx_toolow_row,idx_low] += 1  
                idx_toohigh_row = epi_lengths.sum(1) > n_trials
                idx_toolow_row  = epi_lengths.sum(1) < n_trials
                
            # reloop if necessary
            cond_numtrials = np.all(epi_lengths.sum(1) == n_trials)
            cond_min = np.transpose((epi_lengths < epimin).nonzero()).size == 0
            cond_max = np.transpose((epi_lengths > epimax).nonzero()).size == 0
            if cond_numtrials & cond_min & cond_max:
                redo = False
                break
            if i == i_loop_limit-1:
                # best-case scenario, this condition is never reached
                # added for error detection and future repairs
                print('Episode generator reached limit! Rerunning loop...')
                print(idx_toohigh_row.sum()) # problem: episode number too high
                print(idx_toolow_row.sum()) # problem: episode number too low
            
    # create target vectors for each block
    target_vectors = np.full((n_blocks,n_trials), True)
    # calculate index locations corresponding to target states
    epi_end_locs = epi_lengths.cumsum(1)-1
    # generate boolean vectors corresponding to episode states
    idx_even = np.arange(1,n_episodes,2)
    
    # randomize block starting states (option 1 or 2)
    idx_flip = rng.choice(n_blocks, size=int(round(n_blocks/2)), replace=False) 
    target_vectors[idx_flip,:] = 1-target_vectors[idx_flip,:] 
    
    # apply episodes / state switches
    for j in np.arange(n_blocks):
        even_trials = []
        idx_start = epi_end_locs[j,idx_even-1]+1
        idx_end = epi_end_locs[j,idx_even]+1
    
        # (loop) create array of episode locations to alter
        idx_makefalse = np.array([])
        for i in range(len(idx_start)):
            idx_makefalse = np.append(idx_makefalse, np.arange(idx_start[i],idx_end[i])) 
        idx_makefalse = idx_makefalse.astype(int)
        # (single execution) alter once
        target_vectors[j,idx_makefalse] = ~target_vectors[j,idx_makefalse]
        
    # generate stimulus values
    stim1_vectors = rng.choice(np.arange(-.98,1.,.02), 
                            p=llr_dist, 
                            size=[n_blocks,n_trials], 
                            replace=True)
    
    # apply episode/state switches to stimuli (option 1)
    stim1_vectors[target_vectors] = -stim1_vectors[target_vectors]
    
    # generate stimulus vector for option 2
    stim2_vectors = -stim1_vectors
    
    # convert to torch tensor
    target_vectors = torch.tensor(np.transpose(target_vectors).astype(np.float32))
    stim1_vectors  = torch.tensor(np.transpose(stim1_vectors).astype(np.float32))

    # output stimulus vectors and target vector
    return target_vectors, stim1_vectors, # stim2_vectors

# block stimulus value generator without trial or poisson parameter bounds
def generate_blocks_h(n_trials, n_blocks, hazard=0.1):
    # configuration
    llrmax = 1.7954
    nlevel = 49  # number of stimuli in favor of an option
    
    # calculate probability values
    llr_dist = sample_dist(nlevel, llrmax)
    # generate episode states by using the trial-to-trial hazard rate
    revpt_samp = np.array(rng.random([n_blocks,n_trials]))
    idx_is_revpt = revpt_samp <= hazard
    is_rev = idx_is_revpt.astype(int)
    idx_state = np.cumsum(is_rev, axis=1)
    
    # create target vectors corresponding to the best choice for each block
    target_vectors = idx_state % 2
    
    # randomize block starting states (option 1 or 2)
    idx_flip = rng.choice(n_blocks, size=int(round(n_blocks/2)), replace=False) 
    
    target_vectors[idx_flip,:] = 1-target_vectors[idx_flip,:] 
    
    # generate stimulus values for option 1 (0) assuming that it is the best option
    stim1_vectors = rng.choice(np.arange(-.98,1.,.02), 
                               p=llr_dist, 
                               size=[n_blocks,n_trials], 
                               replace=True)
    
    # apply episode/state switches to option 1 if it was flipped (idx_flip)
    stim1_vectors[target_vectors.astype(bool)] = -stim1_vectors[target_vectors.astype(bool)]
    
    # generate stimulus vector for option 2
    stim2_vectors = -stim1_vectors

    # convert to torch tensor
    target_vectors = torch.tensor(np.transpose(target_vectors).astype(np.float32))
    stim1_vectors  = torch.tensor(np.transpose(stim1_vectors).astype(np.float32))
    
    # output stimulus vectors and target vector
    return target_vectors, stim1_vectors

# Define the RNN model
class CustomRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, noise_value=0.0):
        # bring in methods of nn.Module into RNNModel
        super(CustomRNN, self).__init__()
        # number of units in hidden layer
        self.hidden_dim = hidden_dim
        # noise scalar constant
        self.noise_value = noise_value
        # define layers
        self.i2h = nn.Linear(input_dim, hidden_dim, bias=False) # input to hidden 
        self.h2h = nn.Linear(hidden_dim, hidden_dim, bias=True) # hidden to hidden
        self.h2o = nn.Linear(hidden_dim, output_dim, bias=False) # hidden to output
        # define activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input, hidden_prev, device):
        inp = self.i2h(input).to(device)
        hid = self.h2h(hidden_prev).to(device)
        
        # update hidden state
        combined = inp + hid
        
        # calculate noise
        if self.noise_value != 0:
            noise_samp = torch.normal(mean=torch.zeros(combined.shape, device=device),
                                      std=torch.ones(combined.shape, device=device)*self.noise_value)
        else:
            noise_samp = torch.zeros(combined.shape, device=device)
        
        # calculate hidden state
        hidden = self.tanh(combined + noise_samp)
        hidden.to(device)
        # calculate output
        output = self.sigmoid(self.h2o(hidden))
        output.to(device)

        return output, hidden

    def init_hidden(self, batch_size, device):
        # initialize hidden state
        return torch.zeros(batch_size, self.hidden_dim, requires_grad=True, device=device)

def bayes_inf_model(h, prior, stim):
    hrprior = prior + np.log((1-h)/h + np.exp(-prior)) - np.log((1-h)/h + np.exp(prior))
    return stim*1.7954 + hrprior 
    
# Evaluates the model for training or test
def run_traineval_loop(input_tensor, rnn_model, condition, device):
    input_dims      = input_tensor.shape
    sequence_length = input_dims[0]
    batch_half      = input_dims[1]
    hidden_size     = rnn_model.h2h.in_features
    # preallocate output tensors
    h_activat = torch.full((batch_half, hidden_size, sequence_length), torch.nan, device=device)
    responses = torch.full(input_dims[0:2], torch.nan, device=device)
    respprobs = torch.full(input_dims[0:2], torch.nan, device=device)
    respargmx = torch.full(input_dims[0:2], torch.nan, device=device)
    bayesresp = torch.full(input_dims[0:2], torch.nan, device=device)
    bayespost = torch.full((batch_half, sequence_length), torch.nan, device=device)
    
    # run training loop
    hidden = rnn_model.init_hidden(batch_half, device)
        
    for t in range(sequence_length):
        train_tensor = input_tensor[t,:,:]
        if condition == 'bandit':
            if t == 0: # initialize input
                ''' The RNNs receive information about both bandits on the 1st trial '''
                input_at_t = train_tensor
                input_at_t_argmax = train_tensor
                input_at_t_bayes = train_tensor
                
                ''' The Bayesian inf. model gets the "reward" for option 0 as if it chose it '''
                evidence = input_tensor[t,:,0].squeeze() 

            if t != 0: # get output from previous trial to determine input
                idx = output.unsqueeze(1).to(device) # uncomment later

                # debug something here, where the choices are made by argmax rather than sigmoid
                idx_argmax = respargmx[t-1, :].type(torch.LongTensor).unsqueeze(1)
                
                # gather rewards corresponding to choice, scatter them into correct locations on zero tensor
                input_at_t = torch.zeros(batch_half, 3, device=device).scatter_(1, idx, train_tensor.gather(1, idx))
                input_at_t_argmax = torch.zeros(batch_half, 3, device=device).scatter_(1, idx_argmax, train_tensor.gather(1, idx_argmax))
                
                # sign the reward by the previous choice of Bayes inf model
                idx_bayes = out_bayes.unsqueeze(1).type(torch.int64)
                prevchoice_signed = -2*(bayesresp[t-1, :]-.5)
                input_at_t_bayes = torch.zeros(batch_half, 3, device=device).scatter_(1, idx_bayes, train_tensor.gather(1, idx_bayes))
                evidence = input_at_t_bayes.sum(axis=1)*prevchoice_signed
            
        elif condition == 'fairy':
            input_at_t = train_tensor
            input_at_t_argmax = train_tensor
            # bayesian inf
            input_at_t_bayes = train_tensor
            evidence = input_tensor[t,:,2].squeeze() # bayesinf
        
        if t == 0:
            outcomes = input_at_t_argmax.unsqueeze(2)
            # bayesian inf
            bayesoutc = input_at_t_bayes.unsqueeze(2)
            prior = 0
        else:
            outcomes = torch.cat((outcomes, input_at_t_argmax.unsqueeze(2)), dim=2)
            bayesoutc = torch.cat((bayesoutc, input_at_t_bayes.unsqueeze(2)), dim=2)

        # bayesian inf
        evidence = evidence*1.7954
        posterior = bayes_inf_model(.116, prior, evidence)
        bayespost[:,t] = posterior
        out_bayes = posterior.le(0)
        prior = posterior
        bayesresp[t, :] = out_bayes
        
        output, hidden = rnn_model(input_at_t, hidden, device)  # forward pass
        h_activat[:, :, t] = hidden
        respprobs[t, :] = output.squeeze()  # probability of choosing option 0
        respargmx[t, :] = torch.round(output.squeeze())
        output = torch.bernoulli(output).type(torch.LongTensor).squeeze()
        responses[t, :] = output
        
    out = { 'respprobs': respprobs,
            'responses': responses,
            'h_activat': h_activat,
            'outcomes' : outcomes,
            'bayesresp': bayesresp,
            'bayesoutc': bayesoutc,
            'bayespost': bayespost}
        
    return out

# Creates the input tensor from which bandit or fairy outcomes are taken
def make_three_input_tensor(input_tensor1, condition):
    if condition == 'bandit':
        bandit_opt1 = input_tensor1.unsqueeze(2)
        bandit_opt2 = -bandit_opt1
        output_tensor = torch.cat((bandit_opt2, torch.zeros_like(bandit_opt2)), dim=2)
        output_tensor = torch.cat((bandit_opt1, output_tensor), dim=2)
        
    if condition == 'fairy':
        tensor_shape = input_tensor1.shape
        zero_tensor = torch.zeros(tensor_shape[0], tensor_shape[1], 2)
        output_tensor = torch.cat((zero_tensor, input_tensor1.unsqueeze(2)), dim=2)
        
    return output_tensor

def run_noisyRNN_continue(hidden_size, noise_value, cos_penalty, filepath, run_number):
    # check path and existence of file
    nunit_str = f'{hidden_size:03d}'
    noise_str = f'{int(noise_value*100):03d}'
    cos_str   = f'{int(cos_penalty):03d}'
    run_str   = f'{int(run_number):02d}'
    filename  = f'n{nunit_str}_noise{noise_str}_cosPnlt_{cos_str}_run{run_str}.pckl'

    # open file
    file_dir_full = f'{filepath}/{filename}'
    handle    = open(file_dir_full, 'rb')
    dict_data = pickle.load(handle)
    dict_data['save_dir'] = filepath
    dict_data['run_number'] = int(run_number)

    # continue training from pre-trained data
    run_noisyRNN(hidden_size, noise_value, cos_penalty, dict_data)


def run_noisyRNN(hidden_size=64, noise_value=0.5, cos_penalty=40, previous_saved_data: dict={}):
    # check for continuation option
    is_continuation = False
    if 'weights_dict' in previous_saved_data.keys():
        print(f'Continuing training for run {previous_saved_data['run_number']}...')
        run_number = previous_saved_data['run_number']
        is_continuation = True
        
    # parse argument
    cos_target = 0
    if cos_penalty is None:
        include_cossim_loss = False
    else:
        include_cossim_loss = True
    # RNN:
    input_size  = 3
    output_size = 1
    # Task:
    n_trials   = 72
    n_episodes = 6
    n_blocks   = 200
    # Alignment loss parameters
    cossim_targets = [cos_target]
    # Stopping criterion parameters
    patience = 500  # waiting time after threshold reached for new one
    if noise_value > 0.1:
        patience = patience*2
    if include_cossim_loss == True:
        patience = patience*2
    # Training Parameters
    max_epochs = 50000
    min_epochs = 1000
    if include_cossim_loss == True:
        min_epochs = min_epochs*2
    epochmod   = 200
    # --------------- <end> Constants declaration <end> ---------------
    
    n_units_str = 'n%03d' % hidden_size
    batch_str_arr = ["train","test"]
    cond_str_arr  = ['bandit', 'fairy']
    # Derived constants
    batch_size  = n_blocks # num of "blocks" before gradient descent
    batch_half  = int(round(batch_size/2))
    batch_half_idx = [list(range(0,batch_half)), list(range(batch_half,batch_size))]

    # saving variables
    save_dir = './saved_runs/'
    if is_continuation:
        save_dir = f'./{previous_saved_data['save_dir']}/'
    stop_triggered = False # True if stopping criterion met
    save_initialized = False
    loss_total_min = 1e5

    if not include_cossim_loss:
        cossim_targets = [None]
    cossim_targets = [0.]
    # Error handling
    if n_blocks % 2 != 0:
        n_blocks += 1
        print(f'n_blocks must be an even number greater than 2! Setting n_blocks = {n_blocks}...')
    # Loss function instantiation
    bXEntLoss  = nn.BCELoss()
    normL1Loss = nn.L1Loss() 
    
    print(f"Training with noise value = {noise_value:.2f}...")
    # Begin training loop
    for itgt, tgt in enumerate(cossim_targets): # loop over target cosine similarities
        cossim_target = None
        if include_cossim_loss:
            print(f"Target cosine similarity = {tgt:.2f}. (penalty={cos_penalty})...")
        cossim_target = torch.tensor([tgt])
        
        # set up file names and check for overwrite
        cos_pnlt_str = f"{round(cos_penalty):03d}" if include_cossim_loss else 'Non'
        str_datafile = "%s_noise%03d_cosPnlt_%s_" % (n_units_str, noise_value*100, cos_pnlt_str)

        # begin training operations
        rnn = CustomRNN(input_size, hidden_size, output_size, noise_value) # instantiate model
        optimizer = optim.Adam(rnn.parameters(), lr=0.0005, amsgrad=True, eps=1e-6,) # attaach optimizer to model
        
        # log values for performance and plotting [train, validate]
        if not is_continuation:
            pcor_amx_b  = [[],[]] # argmax [train, test]
            pcor_amx_f  = [[],[]] # argmax [train, test]
            loss_acc_b  = [[],[]] # [train, test]
            loss_acc_f  = [[],[]] # [train, test]
            loss_cos_bf = [[],[]] # [train, test]
            loss_totals = [[],[]] # [train, test]
            weights_dict = {} # hold weights for each run that satisfies saving criterion
        else:
            pcor_amx_b   = previous_saved_data['pcor_amx_b']
            pcor_amx_f   = previous_saved_data['pcor_amx_f']
            loss_acc_b   = previous_saved_data['loss_acc_b']
            loss_acc_f   = previous_saved_data['loss_acc_f']
            loss_cos_bf  = previous_saved_data['loss_cos_bf']
            loss_totals  = previous_saved_data['loss_totals']
            weights_dict = previous_saved_data['weights_dict']
            last_epoch   = list(weights_dict.keys())[-1]
            
            # initialize weights with previously saved weights
            weights_dict = weights_dict[last_epoch]
            rnn.i2h.weight.data = torch.from_numpy(weights_dict['i2h_w'])
            rnn.h2h.weight.data = torch.from_numpy(weights_dict['h2h_w'])
            rnn.h2h.bias.data   = torch.from_numpy(weights_dict['h2h_b'])
            rnn.h2o.weight.data = torch.from_numpy(weights_dict['h2o_w'])
        
        torch.set_grad_enabled(True)
        rnn.to(device)
        
        # loop through epochs
        for epoch in range(max_epochs):
            is_crit_save = False
            # loop over training and validation batches
            for ibatch in range(len(batch_str_arr)):
                if ibatch == 0: # for training
                    optimizer.zero_grad() # reset gradient
                    rnn.train()
                else: # for validation
                    rnn.eval()
                    
                # generate task (hazard rate for training | human task for test)
                if ibatch == 0:
                    out_gen = generate_blocks_h(n_trials, n_blocks, .116) # hazard rate sampled reverals
                else:
                    out_gen = generate_blocks(n_trials, n_episodes, n_blocks) # task generator for HUMAN participants
                target, input_tensor_opt1 = out_gen

                ''' toggle True if the bandit and fairy task deired to be identical '''
                if ibatch == 1 & True:
                    target = target[:,0:batch_half]
                    input_tensor_opt1 = input_tensor_opt1[:,0:batch_half]
                    target = target.repeat(1,2)
                    input_tensor_opt1 = input_tensor_opt1.repeat(1,2)
                
                # allocate variable storage
                respprobs = torch.full(target.shape, torch.nan, device=device)
                respargmx = torch.full(target.shape, torch.nan, device=device)
                
                hidden_actis = torch.full([batch_half, hidden_size, n_trials, len(cond_str_arr)], torch.nan, device=device)
                bayespost = torch.full((n_blocks, n_trials), torch.nan, device=device) # Bayesian model posteri
                
                # loop over bandit [0] or fairy [1] blocks
                for icond, cond_str in enumerate(cond_str_arr):
                    idx_cond = batch_half_idx[icond]
                    
                    input_tensors = make_three_input_tensor(input_tensor_opt1[:,idx_cond], cond_str)
                    out_b = run_traineval_loop(input_tensors, rnn, cond_str, device)
                    respprobs[:,idx_cond]     = out_b["respprobs"]
                    hidden_actis[:,:,:,icond] = out_b["h_activat"]
                    bayespost[idx_cond,:]     = out_b['bayespost'] 
                    respargmx[:,idx_cond] = torch.round(respprobs[:,idx_cond])
                    
                # ---------- Calculate performance metrics ----------
                # concatenate hidden activation tensors for pca
                hidden_actis_b = torch.transpose(hidden_actis[:,:,:,0], 0, 1)
                hidden_actis_f = torch.transpose(hidden_actis[:,:,:,1], 0, 1)
                hid_act_b_concat = torch.reshape(hidden_actis_b,(hidden_size,batch_half*n_trials)).T # samples, features
                hid_act_f_concat = torch.reshape(hidden_actis_f,(hidden_size,batch_half*n_trials)).T # samples, features

                # PCA
                _, _, V_b = torch.pca_lowrank(hid_act_b_concat, q=hidden_size) # V_b: unit dimension, pc dimension
                _, _, V_f = torch.pca_lowrank(hid_act_f_concat, q=hidden_size)

                # calculate all subspace angles for the principal axes in both tasks
                subsp_cos_mat = torch.abs(torch.matmul(V_b.T, V_f)) 
                
                # get bayesian posterior for the two tasks
                bayes_Lb_flat = torch.flatten(bayespost[:batch_half,:]).unsqueeze(1)
                bayes_Lf_flat = torch.flatten(bayespost[batch_half:,:]).unsqueeze(1)

                # Calculate encoding weights using Moore Penrose pseudo-inverse
                X_b = hid_act_b_concat
                y_b = bayes_Lb_flat
                w_enc_b = torch.matmul(torch.matmul(X_b.T, y_b), torch.linalg.pinv(torch.matmul(y_b.T, y_b)))
                X_f = hid_act_f_concat
                y_f = bayes_Lf_flat
                w_enc_f = torch.matmul(torch.matmul(X_f.T, y_f), torch.linalg.pinv(torch.matmul(y_f.T, y_f)))

                # cosine similarity of each PC on the Bayes Optimal Weights
                corr_pc_bayesenc_b = torch.abs(torch.matmul(V_b.T, w_enc_b)) # cosine sim of pcs w/ bayes
                corr_pc_bayesenc_f = torch.abs(torch.matmul(V_f.T, w_enc_f))
                
                # calculate importance weights for cosine similarity of subspace loss 
                cos_wts_add = corr_pc_bayesenc_b.repeat(1, hidden_size) + corr_pc_bayesenc_f.repeat(1, hidden_size).T
                # cos_wts_mul = torch.matmul(corr_pc_bayesenc_b, corr_pc_bayesenc_f.T)
                perc_var_wt = cos_wts_add # * cos_wts_mul
                perc_var_wt = perc_var_wt/perc_var_wt.sum()

                ''' Calculate performance metrics '''
                loss_x_ent = torch.full((2, 1), torch.nan, device=device)
                accuracy = torch.full((2, 1), torch.nan, device=device)
                # LOSS: Binary Cross Entropy (for accuracy)
                for icond, cond_str in enumerate(cond_str_arr):
                    # Accuracy: 
                    with torch.no_grad():
                        accuracy[icond] = torch.mean(torch.eq(respargmx[:,batch_half_idx[icond]],target[:,batch_half_idx[icond]]).float(),0).mean()
                    loss_x_ent[icond] = bXEntLoss(respprobs[:,batch_half_idx[icond]], target[:,batch_half_idx[icond]]) # cross entropy loss

                ''' For saving '''
                # task-wise objective loss
                loss_acc_b[ibatch].append(loss_x_ent[0].item())
                loss_acc_f[ibatch].append(loss_x_ent[1].item())
                # task-wise accuracy
                pcor_amx_b[ibatch].append(accuracy[0].item())
                pcor_amx_f[ibatch].append(accuracy[1].item())
                
                # LOSS: Alignment 
                loss_cos_mat = normL1Loss(subsp_cos_mat, cossim_target.repeat(hidden_size, hidden_size)) # get distance from target for each cosine sim
                loss_cos = (loss_cos_mat * perc_var_wt).sum() # weight the loss and sum 
                loss_cos_bf[ibatch].append(loss_cos.item())
                
                # LOSS: Total
                if include_cossim_loss:
                    loss_total = loss_x_ent.mean() + (loss_cos * cos_penalty)
                else:
                    loss_total = loss_x_ent.mean()
                loss_totals[ibatch].append(loss_total.item())
                
                # backpropagate and optimize (only during training batches)
                if ibatch == 0: 
                    loss_total.backward()
                    optimizer.step()

            # for output logging purposes ONLY
            if not is_continuation:
                true_epoch = epoch
            else:
                true_epoch = epoch + last_epoch 

            # implement stopping check
            if true_epoch > min_epochs: # wait for burn-in period
                if true_epoch == min_epochs + 1:
                    epoch_loss_min = true_epoch
                    
                if ibatch == 1: # evaluate stopping criterion only on test data
                    loss_arr = np.array(loss_totals[1][-patience-1:-1])
                    loss_avg = loss_arr.mean() # running mean 
                    loss_std = loss_arr.std() # running std
                    loss_total_lbound = loss_avg - (3 * loss_std) # the threshold for a loss improvement is 3 stds away from running mean
                    loss_total_lbound = round(loss_total_lbound, 3)
                    
                    if loss_total_lbound < loss_total_min:
                        print(f'New loss minimum achieved ({loss_total_lbound:.3f}) | prev: {loss_total_min:.3f} | (epoch: {true_epoch}) ')
                        epoch_loss_min = epoch
                        loss_total_min = loss_total_lbound 

                if true_epoch >= epoch_loss_min + patience:
                    print('Patience reached. Stopping training and saving to file...')
                    print(f'Patience reached at epoch {true_epoch}')
                    stop_triggered = True
                    
                if epoch == max_epochs:
                    print('Maximum epoch reached. Stopping training and saving to file...')
                    stop_triggered = True

            # Print and log and save
            if (true_epoch % epochmod == 0 and epoch > epochmod-1) or stop_triggered:
                print(f'Current loss: {loss_total.item():.3f} | Accuracy: ({pcor_amx_b[1][-1]:.3f}, {pcor_amx_f[1][-1]:.3f}) | (epoch: {true_epoch})')
                weights_dict[true_epoch] = {
                                'i2h_w' : rnn.i2h.weight.detach().cpu().numpy().copy(),
                                'h2h_w' : rnn.h2h.weight.detach().cpu().numpy().copy(),
                                'h2h_b' : rnn.h2h.bias.detach().cpu().numpy().copy(),
                                'h2o_w' : rnn.h2o.weight.detach().cpu().numpy().copy(), }
                
                stored_vals = { "pcor_amx_b"   : pcor_amx_b, 
                                "pcor_amx_f"   : pcor_amx_f,
                                "loss_acc_b"   : loss_acc_b, 
                                "loss_acc_f"   : loss_acc_f, 
                                "loss_cos_bf"  : loss_cos_bf, 
                                "loss_totals"  : loss_totals, 
                                "weights_dict" : weights_dict,
                                "cos_penalty"  : cos_penalty}

                # maintain only the last 10 sets of weights
                list_epochs = list(stored_vals['weights_dict'].keys())
                if len(list_epochs) > 10:
                    del stored_vals['weights_dict'][list_epochs[0]]

                # save-as-you-go
                if not save_initialized:
                    if not is_continuation:
                        # check in target directory to see if runs already exist to not overwrite
                        list_filenames = glob.glob(f"./{save_dir}/{str_datafile}*.pckl")
                        list_filenames = [name[0:-5] for name in list_filenames]
                        
                        # extract run numbers
                        list_run_nums = [int(name[-2:]) for name in list_filenames]
                        if len(list_run_nums) > 0:
                            last_irun = max(list_run_nums)
                        else:
                            last_irun = -1
                        # set run number
                        irun_file = last_irun+1
                    else:
                        irun_file = run_number
                    run_str = f'run{irun_file:02d}'
                    str_datafile = f'{str_datafile}{run_str}'
                    # turn off run number check
                    save_initialized = True
                
                # save file
                f = open(f'{save_dir+str_datafile}.pckl', 'wb')
                pickle.dump(stored_vals, f)
                f.close()
                print(f'{str_datafile}.pckl saved!')
                print('Operation ongoing...')

            if stop_triggered:
                break
            # </end-of-epoch>
        # </end-of-run>

        # turn off global gradient propagation
        torch.set_grad_enabled(False)
    print('Operation terminated.')
