% Samuel Taylor and Ryan Smith, 2021

% Three-arm bandit fitting script
% Fit a given subject over all 20 blocks
function FinalResults = TAB_fit(subject, data_dir)
    rng('shuffle');
    
    file = [data_dir '/' subject '.csv'];
%     file = [data_dir '/' subject '/T4/behavioral_session/' subject '-T4-_BAN-R1-_BEH.csv'];

    %% 12. Set up model structure
    %==========================================================================
    %==========================================================================

    TpB = 16;     % trials per block
    NB  = 1;     % number of blocks
    N   = TpB*NB; % trials per block * number of blocks
    
    BlockProbs = zeros(NB, 3);
    
    %--------------------------------------------------------------------------

    alpha    = 4;
    cr       = 4;
    beta     = 1;
    eta_win  = 1/2;
    eta_loss = 1/2;
    prior_a  = 1/4;
    omega = .5;

    %% Add Subj Data (Parse the data files)
    subdat = readtable(file); %subject data 
    
    % Parse block structure (probabilities for each bandit within blocks)
    BlockprobData = string(subdat.trial_type(subdat.event_code==3));
    BlockprobData = strrep(BlockprobData,'-',' ');

    for i = 1:NB
        BlockprobData(i,1:3) = strsplit(BlockprobData(i));
    end

    for i = 1:NB
        for j = 1:3
            BlockProbs(i,j) = str2num(BlockprobData(i,j));
        end
    end

    % Parse observations and actions
    sub.o = str2num(cell2mat(subdat.result(subdat.event_code == 5)));
    sub.u = str2num(cell2mat(subdat.response(subdat.event_code == 5)));

    for i = 1:N
        if sub.o(i) == 0 
            sub.o(i) = 3;
        end
         if sub.o(i) == 1 
            sub.o(i) = 2;
        end
    end

    sub.u = sub.u+1;

    o_all = [];
    u_all = [];

    for n = 1:NB
        o_all = [o_all sub.o((n*TpB-(TpB-1)):TpB*n,1)];
        u_all = [u_all sub.u((n*TpB-(TpB-1)):TpB*n,1)];
    end
        
    %% 6.2 Invert model and try to recover original parameters:
    %==========================================================================

    %--------------------------------------------------------------------------
    % This is the model inversion part. Model inversion is based on variational
    % Bayes. The basic idea is to maximise (negative) variational free energy
    % wrt to the free parameters (here: alpha and cr). This means maximising
    % the likelihood of the data under these parameters (i.e., maximise
    % accuracy) and at the same time penalising for strong deviations from the
    % priors over the parameters (i.e., minimise complexity), which prevents
    % overfitting.
    % 
    % You can specify the prior mean and variance of each parameter at the
    % beginning of the TAB_spm_dcm_mdp script.
    %--------------------------------------------------------------------------

    MDP = TAB_gen_model(BlockProbs(1,:),beta,alpha,cr,eta_win,eta_loss,prior_a,omega);

    MDP.BlockProbs = BlockProbs; % Block probabilities
    MDP.TpB        = TpB;        % trials per block
    MDP.NB         = NB;         % number of blocks
    MDP.prior_a    = prior_a;    % prior_a

    DCM.MDP    = MDP;                  % MDP model
    DCM.field  = {'alpha' 'cr' 'eta_win' 'eta_loss' 'prior_a','omega'}; % Parameter field
    DCM.U      = {o_all};              % trial specification (stimuli)
    DCM.Y      = {u_all};              % responses (action)

    DCM        = TAB_inversion(DCM);   % Invert the model

    %% 6.3 Check deviation of prior and posterior means & posterior covariance:
    %==========================================================================

    %--------------------------------------------------------------------------
    % re-transform values and compare prior with posterior estimates
    %--------------------------------------------------------------------------
    field = fieldnames(DCM.M.pE);
    for i = 1:length(field)
        if strcmp(field{i},'eta_win')
            prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
            posterior(i) = 1/(1+exp(-DCM.Ep.(field{i}))); 
        elseif strcmp(field{i},'eta_loss')
            prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
            posterior(i) = 1/(1+exp(-DCM.Ep.(field{i}))); 
        elseif strcmp(field{i},'omega')
            prior(i) = 1/(1+exp(-DCM.M.pE.(field{i})));
            posterior(i) = 1/(1+exp(-DCM.Ep.(field{i}))); 
        else
            prior(i) = exp(DCM.M.pE.(field{i}));
            posterior(i) = exp(DCM.Ep.(field{i}));
        end
    end

    all_MDPs = [];
    
    % Simulate beliefs using fitted values
    for block=1:NB
        if NB ==1
            sim_mdp = TAB_gen_model(BlockProbs(:), 1, posterior(1), posterior(2), posterior(3), posterior(4), posterior(5),posterior(6));
        else
            sim_mdp = TAB_gen_model(BlockProbs(block, :), 1, posterior(1), posterior(2), posterior(3), posterior(4), posterior(5),posterior(6));
        end
        % Deal for all TpB trials within a block
        MDPs(1:TpB) = deal(sim_mdp);
        
        for t=1:TpB
            MDPs(t).o = [1 o_all(t, block); 1 u_all(t, block)];
            MDPs(t).u = u_all(t, block);
        end

        % Run simulation routine
        MDPs  = spm_MDP_VB_X_eta2_forgetting_CMG(MDPs);

        % Save block of MDPs to list of all MDPs
        all_MDPs = [all_MDPs; MDPs'];
        
        clear MDPs;
    end

    % Return input file name, prior, posterior, output DCM structure, and
    % list of MDPs across task using fitted posterior values
    FinalResults = [{file} prior posterior DCM all_MDPs];
end
