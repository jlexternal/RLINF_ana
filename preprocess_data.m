% preprocess_data

clear all
samplename = 'sample2';

for isretest = [0 1]
    filenameadd = '';
    if isretest
        filenameadd = '_retest';
    end
    
    load(sprintf('./constants/constants_rlinf_%s.mat',samplename)); % load nblk, ncond, ntrl, samplename
    load(sprintf('./processed/%s%s/subj_struct_%s%s.mat',samplename,filenameadd, samplename,filenameadd)); % load the raw data structure for data sample
    
    nsubj = numel(subj_struct);
    
    idx_trial   = nan(nsubj,ntrl*nblk*ncnd);
    idx_abstr   = nan(nsubj,ntrl*nblk*ncnd);
    idx_blk     = nan(nsubj,ntrl*nblk*ncnd);
    idx_epi     = nan(nsubj,ntrl*nblk*ncnd);
    idx_cond    = nan(nsubj,ntrl*nblk*ncnd);
    idx_fbabs   = nan(nsubj,ntrl*nblk*ncnd);
    idx_fb      = nan(nsubj,ntrl*nblk*ncnd);
    idx_corr    = nan(nsubj,ntrl*nblk*ncnd);
    idx_left    = nan(nsubj,ntrl*nblk*ncnd);
    idx_blmn    = nan(nsubj,ntrl*nblk*ncnd);
    idx_rt      = nan(nsubj,ntrl*nblk*ncnd);
    idx_bmstate = nan(nsubj,ntrl*nblk*ncnd);
    
    for isubj = 1:nsubj
        if isempty(subj_struct{isubj})
            continue
        end
    
        idx_trial(isubj,:)  = subj_struct{isubj}.itrl; 
        idx_blk(isubj,:)    = subj_struct{isubj}.iblk;
        idx_abstr(isubj,:)  = subj_struct{isubj}.it_abs;
        idx_epi(isubj,:)    = subj_struct{isubj}.iepi;
        idx_cond(isubj,:)   = subj_struct{isubj}.icnd;
        idx_fbabs(isubj,:)  = subj_struct{isubj}.fb;
        idx_fb(isubj,:)     = subj_struct{isubj}.fb_seen;
        idx_corr(isubj,:)   = subj_struct{isubj}.is_corr;
        idx_left(isubj,:)   = subj_struct{isubj}.is_left;
        idx_blmn(isubj,:)   = subj_struct{isubj}.is_blmn;
        idx_rt(isubj,:)     = subj_struct{isubj}.rt;
        idx_bmstate(isubj,:)= subj_struct{isubj}.bm_state;
        
        % add additional trial for each block in idx_trial
        idx_trial(isubj,:) = repmat(1:292,[1 2]);
        
        idx_begin = 1+72*(0:7);
        idxs = nan(size(idx_begin));
        for j = 1:numel(idx_begin)
            idxs(j) = find(idx_abstr(isubj,:) == idx_begin(j),1);
        end
        for i = 1:numel(idxs)
            idx_abstr(isubj,idxs(i):(idxs(i)+72)) = idx_begin(i):(idx_begin(i)+72);
        end
    end
    
    % indicate NaNs where there is no data
    %          |---bandit---| |----fairy----|
    nan_loc = [73 146 219 292 293 366 439 512];
    idx_fbabs(:,nan_loc) = nan; % no actual feedback given on these trials
    idx_fb(:,nan_loc)    = nan; 
    idx_blmn(:,nan_loc(5:end)) = nan; % no actual response given on these trials
    idx_left(:,nan_loc(5:end)) = nan;
    idx_corr(:,nan_loc(5:end)) = nan;
    
    save(sprintf('./processed/%s%s/preprocessed_data_%s%s',samplename,filenameadd,samplename,filenameadd), ...
        'idx_trial','idx_blk','idx_abstr','idx_epi','idx_cond','idx_fbabs','idx_fb','idx_corr','idx_left','idx_blmn','idx_rt','idx_bmstate');
end