import sys
import glob
import pickle
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
sys.path.append('./cluster')
from get_rnn_behavior import (
    get_reversal_behavior,
    get_signed_repetition
)

device = torch.device('cpu') 
# shorthand random Generator
rng = np.random.default_rng()

class Ridge:
    def __init__(self, alpha = 0, fit_intercept = True,):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        
    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        X = X.rename(None)
        y = y.rename(None).view(-1,1)
        assert X.shape[0] == y.shape[0], "Number of X and y rows don't match"
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim = 1)
        # Solving X*w = y with Normal equations:
        # X^{T}*X*w = X^{T}*y 
        lhs = X.T @ X 
        rhs = X.T @ y
        
        if self.alpha == 0:
            self.w = torch.linalg.lstsq(lhs, rhs).solution
        else:
            ridge = self.alpha*torch.eye(lhs.shape[0])
            self.w = torch.linalg.lstsq(lhs + ridge, rhs).solution

    def predict(self, X: torch.tensor) -> None:
        X = X.rename(None)
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1), X], dim = 1)
        return X @ self.w

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
            # bayesian infe
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

def eval_noisyRNN(network_cfg, task_list):
    # parse network cfg
    i2h_w = network_cfg['i2h_w']
    h2h_w = network_cfg['h2h_w']
    h2h_b = network_cfg['h2h_b']
    h2o_w = network_cfg['h2o_w']
    noise_value = network_cfg['noise_value']
    # parse task
    target, input_tensor_opt1 = task_list
    n_trials, n_blocks = input_tensor_opt1.shape
    batch_size  = n_blocks
    batch_half  = int(round(batch_size/2))
    batch_half_idx = [list(range(0,batch_half)), list(range(batch_half,batch_size))]
    # bandit and fairy task identical for condition-wise comparison
    target = target[:,0:batch_half]
    input_tensor_opt1 = input_tensor_opt1[:,0:batch_half]
    target = target.repeat(1,2)
    input_tensor_opt1 = input_tensor_opt1.repeat(1,2)
    
    # RNN cfg
    hidden_size, input_size = i2h_w.shape
    output_size, _ = h2o_w.shape
    # string declarations
    batch_str_arr = ["train","test"]
    cond_str_arr  = ['bandit', 'fairy']
    
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    
    # set up torch for evaluation
    torch.set_grad_enabled(False)
    # instantiate model
    rnn = CustomRNN(input_size, hidden_size, output_size, noise_value) 
    # initialize weights with previously saved weights
    rnn.i2h.weight.data = torch.from_numpy(i2h_w)
    rnn.h2h.weight.data = torch.from_numpy(h2h_w)
    rnn.h2h.bias.data   = torch.from_numpy(h2h_b)
    rnn.h2o.weight.data = torch.from_numpy(h2o_w)
    rnn.to(device)
    rnn.eval()
                
    # allocate variable storage
    respargmx = torch.full(target.shape, torch.nan, device=device)
    bayesresp = torch.full(target.shape, torch.nan, device=device)
    hidden_actis = torch.full([batch_half, hidden_size, n_trials, len(cond_str_arr)], torch.nan, device=device)
    bayespost = torch.full((n_blocks, n_trials), torch.nan, device=device) # Bayesian model posterior
    
    # loop over bandit [0] or fairy [1] blocks
    pcor_argmax = []
    for icond, cond_str in enumerate(cond_str_arr):
        idx_cond = batch_half_idx[icond]
        input_tensors = make_three_input_tensor(input_tensor_opt1[:,idx_cond], cond_str)
        out_b = run_traineval_loop(input_tensors, rnn, cond_str, device)
        hidden_actis[:,:,:,icond] = out_b["h_activat"]
        bayespost[idx_cond,:]     = out_b['bayespost'] 
        bayesresp[:,idx_cond]     = out_b['bayesresp'] 
        respargmx[:,idx_cond]     = torch.round(out_b["respprobs"])
        
    # ---------- Calculate performance metrics ----------
    # concatenate hidden activation tensors for pca
    hidden_actis_b = torch.transpose(hidden_actis[:,:,:,0], 0, 1)
    hidden_actis_f = torch.transpose(hidden_actis[:,:,:,1], 0, 1)
    hid_act_b_concat = torch.reshape(hidden_actis_b,(hidden_size,batch_half*n_trials)).T # samples, features
    hid_act_f_concat = torch.reshape(hidden_actis_f,(hidden_size,batch_half*n_trials)).T # samples, features

    # PCA
    _, S_b, V_b = torch.pca_lowrank(hid_act_b_concat, q=hidden_size) # V_b: unit dimension, pc dimension
    _, S_f, V_f = torch.pca_lowrank(hid_act_f_concat, q=hidden_size)

    # TESTING
    V_b_orthog = torch.matmul(torch.diag(S_b), V_b.T).T
    V_f_orthog = torch.matmul(torch.diag(S_f), V_f.T).T

    # calculate all subspace angles for the principal axes in both tasks
    overlap = torch.matmul(V_b_orthog.T, V_f_orthog)
    subsp_cos_mat = torch.abs(torch.matmul(V_b.T, V_f))
    
    # variance explained by PCs
    S_sq_b = torch.div(S_b**2, hidden_size-1) # eigenvalues of the cov. mat. of hidden_actis_b
    S_sq_sum_b = S_sq_b.sum()
    perc_var_expl_b = torch.div(S_sq_b, S_sq_sum_b) # should be nblock, nunit
    S_sq_f = torch.div(S_f**2, hidden_size-1) # eigenvalues of the cov. mat. of hidden_actis_f
    S_sq_sum_f = S_sq_f.sum()
    perc_var_expl_f = torch.div(S_sq_f, S_sq_sum_f) # should be nblock, nunit
    npc_b = (perc_var_expl_b >= 1/hidden_size).sum()
    npc_f = (perc_var_expl_f >= 1/hidden_size).sum()     

    # TESTING: 
    trace_overlap = torch.trace(subsp_cos_mat) # frob inner prod
    _, S_o, V_o = torch.pca_lowrank(overlap, q=hidden_size)
    S_sq_o = torch.div(S_o**2, hidden_size-1) # eigenvalues of the cov. mat. of hidden_actis_b
    S_sq_sum_o = S_sq_o.sum()
    eigval_weights = torch.div(S_o, S_o.sum()) 
    perc_var_expl_o = torch.div(S_sq_o, S_sq_sum_o) # should be nblock, nunit
    trace_components = torch.diag(subsp_cos_mat)
    trace_weighted = torch.dot(trace_components, eigval_weights) # weighted trace

    # get bayesian posterior for the two tasks
    bayes_Lb_flat = torch.flatten(bayespost[:batch_half,:]).unsqueeze(1)
    bayes_Lf_flat = torch.flatten(bayespost[batch_half:,:]).unsqueeze(1)

    # Calculate ENCODING weights on hidden layer of the Bayes posterior using Moore Penrose pseudo-inverse
    X_b = hid_act_b_concat
    y_b = bayes_Lb_flat
    w_enc_b = torch.matmul(torch.matmul(X_b.T, y_b), torch.linalg.pinv(torch.matmul(y_b.T, y_b)))
    X_f = hid_act_f_concat
    y_f = bayes_Lf_flat
    w_enc_f = torch.matmul(torch.matmul(X_f.T, y_f), torch.linalg.pinv(torch.matmul(y_f.T, y_f)))
    cos_enc_bayes = cos(w_enc_b.T,w_enc_f.T)
    
    # cosine similarity of each PC on the Bayes Optimal Weights
    corr_pc_bayesenc_b = torch.abs(torch.matmul(V_b.T, w_enc_b)) # cosine sim of pcs w/ bayes
    corr_pc_bayesenc_f = torch.abs(torch.matmul(V_f.T, w_enc_f))

    # Calculate DECODING weights on hidden layer of the Bayes posterior using ridge regression
    ridge_alpha = 5
    ridge_reg = Ridge(alpha = ridge_alpha, fit_intercept = True)
    ridge_reg.fit(X_b,y_b)
    w_dec_b = ridge_reg.w[1::]
    L_hat_b = ridge_reg.predict(X_b) # check that regression predicts the posterior (for bandit) - toggle for debugging
    
    ridge_reg.fit(X_f,y_f)
    w_dec_f = ridge_reg.w[1::]
    L_hat_f = ridge_reg.predict(X_f) # check that regression predicts the posterior (for bandit) - toggle for debugging
    cos_dec_bayes = cos(w_dec_b.T,w_dec_f.T)

    # calculate importance weights for cosine similarity of subspace loss 
    cos_wts_add = corr_pc_bayesenc_b.repeat(1, hidden_size) + corr_pc_bayesenc_f.repeat(1, hidden_size).T
    # cos_wts_mul = torch.matmul(corr_pc_bayesenc_b, corr_pc_bayesenc_f.T)
    perc_var_wt = cos_wts_add # * cos_wts_mul
    perc_var_wt = perc_var_wt/perc_var_wt.sum()

    ''' Calculate performance metrics '''
    loss_x_ent = torch.full((2, 1), torch.nan, device=device)
    accuracy = torch.full((2, 1), torch.nan, device=device)
    bayesacc = torch.full((2, 1), torch.nan, device=device)
    for icond, cond_str in enumerate(cond_str_arr):
        # Accuracy: 
        accuracy[icond] = torch.mean(torch.eq(respargmx[:,batch_half_idx[icond]],target[:,batch_half_idx[icond]]).float(),0).mean()
        bayesacc[icond] = torch.mean(torch.eq(bayesresp[:,batch_half_idx[icond]],target[:,batch_half_idx[icond]]).float(),0).mean()

    if False: # suppress text output
        print(f'Accuracy: ({accuracy[0].item():.3f}, {accuracy[1].item():.3f})')
        print(f'Trace: {trace_overlap.item():.4f} N: ({npc_b}, {npc_f})')
        print(f'Trace: {trace_weighted:.4f} (weighted)')
        
        for i in range(hidden_size):
            print(f'{trace_components[i]:.2f} ({eigval_weights[i]:.2f})', end=' ')
            # print(f'{trace_components[i]:.2f} ({perc_var_expl_o[i]:.2f})', end=' ')
        print('')

    out = {
        'accuracy'         : accuracy,        # condition-wise accuracy
        'bayesacc'         : bayesacc,        # condition-wise accuracy for Bayesian model
        'trace_components' : trace_components,# trace components of the CORRELATION matrix between pcs 
        'eig_lin_weights'  : eigval_weights,  # linear weighting
        'eig_quad_weights' : perc_var_expl_o, # quadratic weighting
        'npcs'             : [npc_b, npc_f],  # number of non-random PCs
        'corr_pc_bayesenc' : torch.cat((corr_pc_bayesenc_b, corr_pc_bayesenc_f), 1), 
        'eigvecs_b'         : V_b_orthog,
        'eigvecs_f'         : V_f_orthog,
        'enc_bayes_b'       : w_enc_b,
        'enc_bayes_f'       : w_enc_f,
        'cos_enc_bayes'     : cos_enc_bayes,
        'dec_bayes_b'       : w_dec_b,
        'dec_bayes_f'       : w_dec_f,
        'cos_dec_bayes'     : cos_dec_bayes,
        
    }
    return out
    
            


















