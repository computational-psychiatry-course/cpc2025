

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-- Simplified Simulation Script --%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Supplementary Code for: A Step-by-Step Tutorial on Active Inference Modelling and its 
% Application to Empirical Data

% By: Ryan Smith, Karl J. Friston, Christopher J. Whyte

rng('shuffle')
close all
clear

% Add paths to SPM
addpath('C:\Users\CGoldman\spm12\')
addpath('C:\Users\CGoldman\spm12\toolbox\DEM\')

% This code simulates a single trial of the explore-exploit task introduced 
% in the active inference tutorial using a stripped down version of the model
% inversion scheme implemented in the spm_MDP_VB_X.m script. 

% Note that this implementation uses the marginal message passing scheme
% described in (Parr et al., 2019), and will return very slightly 
% (negligably) different values than the spm_MDP_VB_X.m script in 
% simulation results.

% Parr, T., Markovic, D., Kiebel, S., & Friston, K. J. (2019). Neuronal 
% message passing using Mean-field, Bethe, and Marginal approximations. 
% Scientific Reports, 9, 1889.

%% Simulation Settings

% To simulate the task when prior beliefs (d) are separated from the 
% generative process, set the 'Gen_model' variable directly
% below to 1. To do so for priors (d), likelihoods (a), and habits (e), 
% set the 'Gen_model' variable to 2:

Gen_model = 2; % as in the main tutorial code, many parameters can be adjusted
               % in the model setup, within the
               % presentation_hint_task_gen_model_CMG
               % function starting on line 810. This includes, among
               % others (similar to in the main tutorial script):

% prior beliefs about context (d): alter line 866

% beliefs about hint accuracy in the likelihood (a): alter lines 986-988

% to adjust habits (e), alter line 1145

%% Specify Generative Model

MDP = presentation_hint_task_gen_model_blank(Gen_model);


% Model specification is reproduced at the bottom of this script (starting 
% on line 810), but see main tutorial script for more complete walk-through

%% Model Inversion to Simulate Behavior
%==========================================================================

% Normalize generative process and generative model
%--------------------------------------------------------------------------

% before sampling from the generative process and inverting the generative 
% model we need to normalize the columns of the matrices so that they can 
% be treated as a probability distributions

% generative process
A = MDP.A;         % Likelihood matrices
B = MDP.B;         % Transition matrices
C = MDP.C;         % Preferences over outcomes
D = MDP.D;         % Priors over initial states    
T = MDP.T;         % Time points per trial
V = MDP.V;         % Policies
beta = MDP.beta;   % Expected free energy precision
alpha = MDP.alpha; % Action precision
eta = MDP.eta;     % Learning rate
omega = MDP.omega; % Forgetting rate

A = col_norm(A);
B = col_norm(B);
D = col_norm(D);

% generative model (lowercase matrices/vectors are beliefs about capitalized matrices/vectors)

NumPolicies = MDP.NumPolicies; % Number of policies
NumFactors = MDP.NumFactors;   % Number of state factors

% Store initial paramater values of generative model for free energy 
% calculations after learning
%--------------------------------------------------------------------------

% 'complexity' of d vector concentration paramaters
if isfield(MDP,'d')
    for factor = 1:numel(MDP.d)
        % store d vector values before learning
        d_prior{factor} = MDP.d{factor};
        % compute "complexity" by subtracting the inverse of the matrix values from
        % the inverse of the sum of the values. Lower concentration paramaters have
        % smaller (more negative) values creating a lower expected free energy thereby
        % encouraging 'novel' behaviour 
        d_complexity{factor} = spm_wnorm(d_prior{factor});
    end 
end 

if isfield(MDP,'a')
    % complexity of a maxtrix concentration parameters
    % Why do we need to zero-out complexity for elements with value zero
    % here but not in d_complexity?
    for modality = 1:numel(MDP.a)
        a_prior{modality} = MDP.a{modality};
        a_complexity{modality} = spm_wnorm(a_prior{modality}).*(a_prior{modality} > 0);
    end
end  

% Normalise matrices before model inversion/inference
%--------------------------------------------------------------------------

% normalize A matrix: Likelihood 
if isfield(MDP,'a')
    a = col_norm(MDP.a);
else 
    a = col_norm(MDP.A);
end 

% normalize B matrix: Transition probabilities
if isfield(MDP,'b')
    b = col_norm(MDP.b);
else 
    b = col_norm(MDP.B);
end 

% normalize C (preference distribution) and transform into log probability
% Notice how preference distribution becomes less precise from T=2 to T=3
% for outcome modality 2.
% Iterate through outcome modalities:
for ii = 1:numel(C)
    C{ii} = MDP.C{ii} + 1/32;
    for t = 1:T
        C{ii}(:,t) = nat_log(exp(C{ii}(:,t))/sum(exp(C{ii}(:,t))));
    end 
end 

% normalize D vector (prior)
if isfield(MDP,'d')
    d = col_norm(MDP.d);
else 
    d = col_norm(MDP.D);
end 

% normalize E vector (habits)
if isfield(MDP,'e')
    E = MDP.e;
    E = E./sum(E);
elseif isfield(MDP,'E')
    E = MDP.E;
    E = E./sum(E);
else
    E = col_norm(ones(NumPolicies,1));
    E = E./sum(E);
end

% Initialize variables
%--------------------------------------------------------------------------

% numbers of transitions, policies and states
NumModalities = numel(a);                    % number of outcome factors
NumFactors = numel(d);                       % number of hidden state factors
NumPolicies = size(V,2);                     % number of allowable policies
for factor = 1:NumFactors
    NumStates(factor) = size(b{factor},1);   % number of hidden states
    NumControllable_transitions(factor) = size(b{factor},3); % number of hidden controllable hidden states for each factor (number of B matrices i.e., size of dim 3)
end

% initialize the approximate posterior over states conditioned on policies
% for each factor as a flat distribution over states at each time point
% Row for each state within the factor
% Col for each time point
% Matrix for each policy

for policy = 1:NumPolicies
    for factor = 1:NumFactors
        NumStates(factor) = length(D{factor}); % number of states in each hidden state factor
        state_posterior{factor} = ones(NumStates(factor),T,policy)/NumStates(factor); 
    end  
end 

% initialize the approximate posterior over policies as a flat distribution 
% over policies at each time point
% Row for each policy
% Col for each time point
policy_posteriors = ones(NumPolicies,T)/NumPolicies; 

% initialize posterior over actions
% Col for each time point in which an action can be taken
% Row for each state factor
chosen_action = zeros(ndims(B),T-1);
    
% Since there is only one policy, fill in the top row of the chosen_action
% matrix because it is a non-controllable transition (people cannot change
% the context state factor)
for factors = 1:NumFactors 
    if NumControllable_transitions(factors) == 1
        chosen_action(factors,:) = ones(1,T-1);
    end
end
MDP.chosen_action = chosen_action;

% initialize expected free energy precision (beta)
posterior_beta = 1;
gamma(1) = 1/posterior_beta; % expected free energy precision
    
% message passing variables
TimeConst = 4; % time constant for gradient descent
NumIterations  = 16; % number of message passing iterations

% store epistemic,pragmatic,novelty value per policy per timestep
G_epistemic_per_timestep = zeros(NumPolicies, T,T);  % (policy, timestep, timepoint)
G_novelty_per_timestep = zeros(NumPolicies, T,T);  % (policy, timestep, timepoint)
G_pragmatic_per_timestep = zeros(NumPolicies, T,T);  % (policy, timestep, timepoint)


% Lets go! Message passing and policy selection 
%--------------------------------------------------------------------------

for t = 1:T % loop over time points  
    
    % sample generative process
    %----------------------------------------------------------------------
    
    for factor = 1:NumFactors % number of hidden state factors
        % Here we sample from the prior distribution over states to obtain the
        % state at each time point. At T = 1 we sample from the D vector, and at
        % time T > 1 we sample from the B matrix. To do this we make a vector 
        % containing the cumulative sum of the columns (which we know sum to one), 
        % generate a random number (0-1),and then use the find function to take 
        % the first number in the cumulative sum vector that is >= the random number. 
        % For example if our D vector is [.5 .5] 50% of the time the element of the 
        % vector corresponding to the state one will be >= to the random number. 

        % sample states 
        if t == 1
            % Use D Matrix to get current state
            initial_state = "TODO"; 
        elseif t>1
            % Use B Matrix to transition from previous state to current
            % state based on action at previous time step
            initial_state = B{factor}(:,true_states(factor,t-1),MDP.chosen_action(factor,t-1));
        end 
        true_states(factor,t) = find(cumsum(initial_state)>= rand,1);
    end 
    % sample observations; Carter changed from a to A
    % Remember that first modality is observe no hint/left hint/right
    % hint, second is null/lose/win, third is start/get hint/choose
    % left/choose right
    for modality = 1:NumModalities % loop over number of outcome modalities
        outcomes(modality,t) = find(cumsum(A{modality }(:,true_states(1,t),true_states(2,t)))>=rand,1);
    end

    % express observations as a structure containing a 1 x observations 
    % vector for each modality with a 1 in the position corresponding to
    % the observation recieved on that trial
    for modality = 1:NumModalities
        vec = zeros(1,size(a{modality},1));
        index = outcomes(modality,t);
        vec(1,index) = 1;
        O{modality,t} = vec;
        clear vec
    end 
    
    % marginal message passing (minimize F and infer posterior over states)
    %----------------------------------------------------------------------
    
    for policy = 1:NumPolicies
        for Ni = 1:NumIterations % number of iterations of message passing  
            for factor = 1:NumFactors
            lnAo = zeros(size(state_posterior{factor})); % initialise matrix containing the log likelihood of observations
                for tau = 1:T % loop over tau
                    if tau<t+1 % Collect an observation from the generative process when tau <= t
                        % Carter added factor as an argument because
                        % get_likelihood appeared to need it
                        lnAo = get_likelihood(lnAo, NumModalities, a, outcomes, tau, state_posterior, NumFactors,factor);
                    end
                    % 'forwards' and 'backwards' messages at each tau
                    if tau == 1 % first tau
                        lnD = nat_log(d{factor}); 
                        lnBs = nat_log(B_norm(b{factor}(:,:,V(tau,policy,factor))')*state_posterior{factor}(:,tau+1,policy));% backward message
                    elseif tau == T % last tau                    
                        lnD  = nat_log((b{factor}(:,:,V(tau-1,policy,factor)))*state_posterior{factor}(:,tau-1,policy));% forward message 
                        lnBs = zeros(size(d{factor})); % no backward message (no contribution from future)
                    else % 1 > tau > T
                        lnD  = nat_log(b{factor}(:,:,V(tau-1,policy,factor))*state_posterior{factor}(:,tau-1,policy));% forward message
                        lnBs = "TODO"; % backward message
                    end

                    v_depolarization = nat_log(state_posterior{factor}(:,tau,policy)); % convert approximate posteriors into depolarisation variable v 
                    % here we both combine the messages and perform a gradient
                    % descent on the posterior 
                    v_depolarization = v_depolarization + (.5*lnD + .5*lnBs + lnAo(:,tau) - v_depolarization)/TimeConst;
                    % variational free energy at each time point
                    % The line below is slightly different from https://github.com/rssmith33/Active-Inference-Tutorial-Scripts/blob/main/Simplified_simulation_script.m
                    % This is canonical free energy; smaller is better
                    Ft(tau,Ni,t,factor) = state_posterior{factor}(:,tau,policy)' * ...
                        ( nat_log(state_posterior{factor}(:,tau,policy)) - (lnAo(:,tau) + 0.5*lnD + 0.5*lnBs) );
                    % update posterior by running v through a softmax 
                    state_posterior{factor}(:,tau,policy) = (exp(v_depolarization)/sum(exp(v_depolarization)));    
                    % store state_posterior (normalised firing rate) from each epoch of
                    % gradient descent for each tau
                    normalized_firing_rates{factor}(Ni,:,tau,t,policy) = state_posterior{factor}(:,tau,policy);                   
                    % store v (non-normalized log posterior or 'membrane potential') 
                    % from each epoch of gradient descent for each tau
                    prediction_error{factor}(Ni,:,tau,t,policy) = v_depolarization;
                    clear v
                end
            end
        end        
      % variational free energy for each policy (F)
      Fintermediate = sum(Ft,4); % sum over hidden state factors (Fintermediate is an intermediate F value)
      Fintermediate = squeeze(sum( Fintermediate,1)); % sum over tau and squeeze into 16x1 matrix
      % store variational free energy at last iteration of message passing
      F(policy,t) = Fintermediate(end);
      clear Fintermediate
    end 

    % expected free energy (G) under each policy
    %----------------------------------------------------------------------
    
    % initialize intermediate expected free energy variable (Gintermediate) for each policy
    Gintermediate = zeros(NumPolicies,1);  
    % policy horizon for 'counterfactual rollout' for deep policies (described below)
    horizon = T;

    % loop over policies
    for policy = 1:NumPolicies
        
        % Bayesian surprise about 'd'
        if isfield(MDP,'d')
            for factor = 1:NumFactors
                Gintermediate(policy) = Gintermediate(policy) - d_complexity{factor}'*state_posterior{factor}(:,1,policy);
                % save novelty value for this policy/timestep/timepoint
                G_novelty_per_timestep(policy,:,t) = -d_complexity{factor}'*state_posterior{factor}(:,1,policy);
            end 
        end
         
        % This calculates the expected free energy from time t to the
        % policy horizon which, for deep policies, is the end of the trial T.
        % We can think about this in terms of a 'counterfactual rollout'
        % that asks, "what policy will best resolve uncertainty about the 
        % mapping between hidden states and observations (maximize
        % epistemic value) and bring about preferred outcomes"?
   
        for timestep = t:horizon
            % grab expected states for each policy and time
            for factor = 1:NumFactors
                Expected_states{factor} = state_posterior{factor}(:,timestep,policy);
            end 
            
            % calculate epistemic value term (Bayesian Surprise) and add to
            % expected free energy. More positive value means greater
            % reduction in entropy
            Gintermediate(policy) = Gintermediate(policy) + G_epistemic_value(a(:),Expected_states(:));
            % save epistemic value for this policy/timestep/timepoint
            G_epistemic_per_timestep(policy,timestep,t) = G_epistemic_value(a(:),Expected_states(:));

            for modality = 1:NumModalities
                % prior preferences about outcomes
                predictive_observations_posterior = cell_md_dot(a{modality},Expected_states(:)); %posterior over observations at timestep=tau under this policy
                Gintermediate(policy) = Gintermediate(policy) + predictive_observations_posterior'*(C{modality}(:,timestep)); % Remember that first modality is hint, second is outcome, third is action
                                
                % save pragmatic value for this policy/timestep/timepoint
                G_pragmatic_per_timestep(policy,timestep,t) = G_pragmatic_per_timestep(policy,timestep,t) + predictive_observations_posterior'*(C{modality}(:,timestep));

                % Bayesian surprise about parameters 
                % Determines how much likelihood matrix will change with
                % the observation expected under this policy
                if isfield(MDP,'a')
                    Gintermediate(policy) = Gintermediate(policy) - cell_md_dot(a_complexity{modality},{predictive_observations_posterior Expected_states{:}});
                    % Save novelty value for this policy/timestep/timepoint
                    G_novelty_per_timestep(policy,timestep,t) = G_novelty_per_timestep(policy,timestep,t) - cell_md_dot(a_complexity{modality},{predictive_observations_posterior Expected_states{:}});
                end
            end 
        end 
    end 
    

    % store expected free energy for each time point and clear intermediate
    % variable
    G(:,t) = Gintermediate;
    clear Gintermediate
    
    % infer policy, update precision and calculate BMA over policies
    %----------------------------------------------------------------------
    

    % loop over policy selection using variational updates to gamma to
    % estimate the optimal contribution of expected free energy to policy
    % selection. This has the effect of down-weighting the contribution of 
    % variational free energy to the posterior over policies when the 
    % difference between the prior and posterior over policies is large (as
    % if to say that our expected free energy is unreliable so we
    % should go with habit instead)
    
    if t > 1
        gamma(t) = gamma(t - 1);
    end
    for ni = 1:Ni 
        % posterior and prior over policies        
        policy_priors(:,t) = exp(log(E) + gamma(t)*G(:,t))/sum(exp(log(E) + gamma(t)*G(:,t)));% prior over policies
        policy_posteriors(:,t) = "TODO"; % posterior over policies 
        
        % expected free energy precision (beta)
        G_error = (policy_posteriors(:,t) - policy_priors(:,t))'*G(:,t);
        beta_update = posterior_beta - beta + G_error; % free energy gradient w.r.t gamma
        posterior_beta = posterior_beta - beta_update/2; 
        gamma(t) = 1/posterior_beta;
        
        % simulate dopamine responses
        n = (t - 1)*Ni + ni;
        gamma_update(n,1) = gamma(t); % simulated neural encoding of precision (posterior_beta^-1)
                                      % at each iteration of variational updating
        policy_posterior_updates(:,n) = policy_posteriors(:,t); % neural encoding of policy posteriors
        policy_posterior(1:NumPolicies,t) = policy_posteriors(:,t); % record posterior over policies 

    end 
    
    % bayesian model average of hidden states (averaging over policies)
    for factor = 1:NumFactors
        for tau = 1:T
            % reshape state_posterior into a matrix of size NumStates(factor) x NumPolicies and then dot with policies
            BMA_states{factor}(:,tau) = reshape(state_posterior{factor}(:,tau,:),NumStates(factor),NumPolicies)*policy_posteriors(:,t);
        end
    end
    
    % action selection
    %----------------------------------------------------------------------
    
    % The probability of emitting each particular action is a softmax function 
    % of a vector containing the probability of each action summed over 
    % each policy. E.g. if there are three policies, a posterior over policies of 
    % [.4 .4 .2], and two possible actions, with policy 1 and 2 leading 
    % to action 1, and policy 3 leading to action 2, the probability of 
    % each action is [.8 .2]. This vector is then passed through a softmax function 
    % controlled by the inverse temperature parameter alpha which by default is extremely 
    % large (alpha = 512), leading to deterministic selection of the action with 
    % the highest probability. 
    
    if t < T

        % marginal posterior over action (for each factor)
        action_posterior_intermediate = zeros([NumControllable_transitions(end),1])';

        for policy = 1:NumPolicies % loop over number of policies
            sub = num2cell(V(t,policy,:));
            action_posterior_intermediate(sub{:}) = action_posterior_intermediate(sub{:}) + policy_posteriors(policy,t);
        end
        
        % action selection (softmax function of action potential)
        sub = repmat({':'},1,NumFactors);
        
        action_posterior_intermediate(:) = "TODO";
        
        action_posterior(sub{:},t) = action_posterior_intermediate;

        % next action - sampled from marginal posterior
        ControlIndex = find(NumControllable_transitions>1);
        action = (1:1:NumControllable_transitions(ControlIndex)); % 1:number of control states
        for factors = 1:NumFactors 
            if NumControllable_transitions(factors) > 2 % if there is more than one control state
                ind = find(rand < cumsum(action_posterior_intermediate(:)),1);  
                MDP.chosen_action(factor,t) = action(ind);
            end
        end

    end % end of state and action selection   
         
end % end loop over time points

% accumulate concentration paramaters (learning)
%--------------------------------------------------------------------------

for t = 1:T
    % a matrix (likelihood)
    if isfield(MDP,'a')
        for modality = 1:NumModalities
            a_learning = O(modality,t)';
            for  factor = 1:NumFactors
                a_learning = spm_cross(a_learning,BMA_states{factor}(:,t));
            end
            a_learning = a_learning.*(MDP.a{modality} > 0);
            MDP.a{modality} = (MDP.a{modality}-MDP.a_0{modality})*(1-omega) + MDP.a_0{modality} + a_learning*eta;
            %MDP.a{modality} = MDP.a{modality}*omega + a_learning*eta;
        end
    end 
end 
 
% initial hidden states d (priors):
if isfield(MDP,'d')
    for factor = 1:NumFactors
        i = MDP.d{factor} > 0;
        MDP.d{factor}(i) = "TODO"; % Hint: the variable BMA_states{factor}(i,1) represents what was "learned".
    end
end

% policies e (habits)
if isfield(MDP,'e')
    MDP.e = (1-omega)*(MDP.e - MDP.e_0) + MDP.e_0 + eta*policy_posterior(:,T);
    %MDP.e = omega*MDP.e + eta*policy_posterior(:,T);
end


% Free energy of concentration parameters
%--------------------------------------------------------------------------

% Here we calculate the KL divergence (negative free energy) of the concentration 
% parameters of the learned distribution before and after learning has occured on 
% each trial. 

% (negative) free energy of a
for modality = 1:NumModalities
    if isfield(MDP,'a')
        MDP.Fa(modality) = - spm_KL_dir(MDP.a{modality},a_prior{modality});
    end
end

% (negative) free energy of d
for factor = 1:NumFactors
    if isfield(MDP,'d')
        MDP.Fd(factor) = - spm_KL_dir(MDP.d{factor},d_prior{factor});
    end
end

% (negative) free energy of e
if isfield(MDP,'e')
    MDP.Fe = - spm_KL_dir(MDP.e,E);
end

% simulated dopamine responses (beta updates)
%----------------------------------------------------------------------
% "deconvolution" of neural encoding of precision
if NumPolicies > 1
    phasic_dopamine = 8*gradient(gamma_update) + gamma_update/8;
else
    phasic_dopamine = [];
    gamma_update = [];
end

% Bayesian model average of neuronal variables; normalized firing rate and
% prediction error
%----------------------------------------------------------------------
for factor = 1:NumFactors
    BMA_normalized_firing_rates{factor} = zeros(Ni,NumStates(factor),T,T);
    BMA_prediction_error{factor} = zeros(Ni,NumStates(factor),T,T);
    for t = 1:T
        for policy = 1:NumPolicies 
            %normalised firing rate
            BMA_normalized_firing_rates{factor}(:,:,1:T,t) = BMA_normalized_firing_rates{factor}(:,:,1:T,t) + normalized_firing_rates{factor}(:,:,1:T,t,policy)*policy_posterior(policy,t);
            %depolarisation
            BMA_prediction_error{factor}(:,:,1:T,t) = BMA_prediction_error{factor}(:,:,1:T,t) + prediction_error{factor}(:,:,1:T,t,policy)*policy_posterior(policy,t);
        end
    end
end

% store variables in MDP structure
%----------------------------------------------------------------------

MDP.T  = T;                                   % number of belief updates
MDP.O  = O;                                   % outcomes
MDP.P  = action_posterior;                    % probability of action at time 1,...,T - 1
MDP.R  = policy_posterior;                    % Posterior over policies
MDP.Q  = state_posterior(:);                  % conditional expectations over N states
MDP.X  = BMA_states(:);                       % Bayesian model averages over T outcomes
MDP.C  = C(:);                                % preferences
MDP.G  = G;                                   % expected free energy
MDP.F  = F;                                   % variational free energy

MDP.s = true_states;                          % states
MDP.o = outcomes;                             % outcomes
MDP.u = MDP.chosen_action;                    % actions

MDP.w  = gamma;                               % posterior expectations of expected free energy precision (gamma)
MDP.vn = BMA_prediction_error(:);             % simulated neuronal prediction error
MDP.xn = BMA_normalized_firing_rates(:);      % simulated neuronal encoding of hidden states
MDP.un = policy_posterior_updates;            % simulated neuronal encoding of policies
MDP.wn = gamma_update;                        % simulated neuronal encoding of policy precision (beta)
MDP.dn = phasic_dopamine;                     % simulated dopamine responses (deconvolved)

%% Plot
%==========================================================================

% trial behaviour
spm_figure('GetWin','Figure 1'); clf    % display behavior
spm_MDP_VB_trial(MDP); 

% neuronal responces
spm_figure('GetWin','Figure 2'); clf    % display behavior
spm_MDP_VB_LFP(MDP,[],1); 

%% Functions
%==========================================================================

% normalise vector columns
function b = col_norm(B)
    numfactors = numel(B);
    for f = 1:numfactors
        bb{f} = B{f}; 
        z = sum(bb{f},1); %create normalizing constant from sum of columns
        bb{f} = bb{f}./z; %divide columns by constant
    end 
    b = bb;
end 

% norm the elements of B transpose as required by MMP
function b = B_norm(B)
bb = B; 
z = sum(bb,1); %create normalizing constant from sum of columns
bb = bb./z; % divide columns by constant
bb(isnan(bb)) = 0; %replace NaN with zero
b = bb;
% insert zero value condition
end 

% natural log that replaces zero values with very small values for numerical reasons.
function y = nat_log(x)
y = log(x+exp(-16));
end 

% dot product along dimension f
function B = md_dot(A,s,f)
if f == 1
    B = A'*s;
elseif f == 2
    B = A*s;
end 
end


%--- SPM functions
%==========================================================================

% These functions have been replicated (with permission) from the spm
% toolbox. To aid in understading, some variable names have been changed.

function X = cell_md_dot(X,x)
% initialize dimensions
DIM = (1:numel(x)) + ndims(X) - numel(x);

% compute dot product using recursive sums (and bsxfun)
for d = 1:numel(x)
    s         = ones(1,ndims(X));
    s(DIM(d)) = numel(x{d});
    X         = bsxfun(@times,X,reshape(full(x{d}),s));
    X         = sum(X,DIM(d));
end

% eliminate singleton dimensions
X = squeeze(X);
end 

function lnAo = get_likelihood(lnAo, NumModalities, a, outcomes, tau, state_posterior, NumFactors,factor)
    for modal = 1:NumModalities % loop over observation modalities
        % this line uses the observation at each tau to index
        % into the A matrix to grab the likelihood of each hidden state
        lnA = permute(nat_log(a{modal}(outcomes(modal,tau),:,:,:,:,:)),[2 3 4 5 6 1]);                           
        for fj = 1:NumFactors
            % dot product with state vector from other hidden state factors 
            % (this is what allows hidden states to interact in the likleihood mapping)    
            if fj ~= factor        
                lnAs = md_dot((lnA),state_posterior{fj}(:,tau),fj);
                clear lnA
                lnA = lnAs; 
                clear lnAs
            end
        end
        lnAo(:,tau) = lnAo(:,tau) + lnA;
    end
end
% epistemic value term (Bayesian surprise) in expected free energy 
function G = G_epistemic_value(A,s)
    
% auxiliary function for Bayesian suprise or mutual information
% FORMAT [G] = spm_MDP_G(A,s)
%
% A   - likelihood array (probability of outcomes given causes)
% s   - probability density of causes

% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_MDP_G.m 7306 2018-05-07 13:42:02Z karl $

% probability distribution over the hidden causes: i.e., q(s)
qx = spm_cross(s); % this is the outer product of the posterior over states
                   % calculated with respect to itself. In essence, it
                   % turns the marginal distribution over states for each
                   % factor into a joint distribution over all states

% accumulate expectation of entropy
G     = 0;
qo    = 0; % q(o) =  E_{q(s)}[p(o|s)] 
for i = find(qx > exp(-16))'
    % probability over outcomes for this combination of causes
    po   = 1;
    for g = 1:numel(A)
        po = spm_cross(po,A{g}(:,i));
    end
    po = po(:); % po = p(o|s)
    qo = qo + qx(i)*po; % qo = q(o) = E_{q(s)}[p(o|s)] 
    G  = G  + qx(i)*po'*nat_log(po); % E_{q(s)} E_{p(o|s)}[ ln p(o|s) ]
end

% subtract entropy of expected observations to calculate mutual information
% Result is 
%   G = E_{q(s)} E_{p(o|s)}[ ln p(o|s) ] - E_{q(o)}[lnP(o)]
%   G = E_{q(s)}[ KL( p(o|s) || q(o) ) ]
%   G = E_{q(o)}[ KL( q(s|o) || q(s) ) ] 

G  = G - qo'*nat_log(qo);

% Epistemic value = mutual information between states and outcomes
% i.e., reduction in state uncertainty when outcomes are observed or reduction of 
% outcome uncertainty when states are observed.
    
end 

%--------------------------------------------------------------------------
function A  = spm_wnorm(A)
% This uses the bsxfun function to subtract the inverse of each column
% entry from the inverse of the sum of the columns and then divide by 2.
% 
A   = A + exp(-16);
A   = bsxfun(@minus,1./sum(A,1),1./A)/2;
end 

function sub = spm_ind2sub(siz,ndx)
% subscripts from linear index
% 

n = numel(siz);
k = [1 cumprod(siz(1:end-1))];
for i = n:-1:1
    vi       = rem(ndx - 1,k(i)) + 1;
    vj       = (ndx - vi)/k(i) + 1;
    sub(i,1) = vj;
    ndx      = vi;
end
end 

%--------------------------------------------------------------------------
function [Y] = spm_cross(X,x,varargin)
% Multidimensional outer product
% FORMAT [Y] = spm_cross(X,x)
% FORMAT [Y] = spm_cross(X)
%
% X  - numeric array
% x  - numeric array
%
% Y  - outer product
%
% See also: spm_dot
% Copyright (C) 2015 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_cross.m 7527 2019-02-06 19:12:56Z karl $

% handle single inputs
if nargin < 2
    if isnumeric(X)
        Y = X;
    else
        Y = spm_cross(X{:});
    end
    return
end

% handle cell arrays

if iscell(X), X = spm_cross(X{:}); end
if iscell(x), x = spm_cross(x{:}); end

% outer product of first pair of arguments (using bsxfun)
A = reshape(full(X),[size(X) ones(1,ndims(x))]);
B = reshape(full(x),[ones(1,ndims(X)) size(x)]);
Y = squeeze(bsxfun(@times,A,B));

% and handle remaining arguments
for i = 1:numel(varargin)
    Y = spm_cross(Y,varargin{i});
end
end 

%--------------------------------------------------------------------------
function [d] = spm_KL_dir(q,p)
% KL divergence between two Dirichlet distributions
% FORMAT [d] = spm_kl_dirichlet(lambda_q,lambda_p)
%
% Calculate KL(Q||P) = <log Q/P> where avg is wrt Q between two Dirichlet 
% distributions Q and P
%
% lambda_q   -   concentration parameter matrix of Q
% lambda_p   -   concentration parameter matrix of P
%
% This routine uses an efficient computation that handles arrays, matrices 
% or vectors. It returns the sum of divergences over columns.
%
% see also: spm_kl_dirichlet.m (for rwo vectors)
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% Will Penny 
% $Id: spm_KL_dir.m 7382 2018-07-25 13:58:04Z karl $

%  KL divergence based on log beta functions
d = spm_betaln(p) - spm_betaln(q) - sum((p - q).*spm_psi(q + 1/32),1);
d = sum(d(:));

return

% check on KL of Dirichlet ditributions
p  = rand(6,1) + 1;
q  = rand(6,1) + p;
p0 = sum(p);
q0 = sum(q);

d  = q - p;
KL = spm_betaln(p) - spm_betaln(q) + d'*spm_psi(q)
kl = gammaln(q0) - sum(gammaln(q)) - gammaln(p0) + sum(gammaln(p)) + ...
    d'*(spm_psi(q) - spm_psi(q0))
end 

%--------------------------------------------------------------------------
function y = spm_betaln(z)
% returns the log the multivariate beta function of a vector.
% FORMAT y = spm_betaln(z)
%   y = spm_betaln(z) computes the natural logarithm of the beta function
%   for corresponding elements of the vector z. if concerned is an array,
%   the beta functions are taken over the elements of the first to mention
%   (and size(y,1) equals one).
%
%   See also BETAINC, BETA.
%   Ref: Abramowitz & Stegun, Handbook of Mathematical Functions, sec. 6.2.
%   Copyright 1984-2004 The MathWorks, Inc. 

% Copyright (C) 2005 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_betaln.m 7508 2018-12-21 09:49:44Z thomas $

% log the multivariate beta function of a vector
if isvector(z)
    z     = z(find(z)); %#ok<FNDSB>
    y     = sum(gammaln(z)) - gammaln(sum(z));
else
    for i = 1:size(z,2)
        for j = 1:size(z,3)
            for k = 1:size(z,4)
                for l = 1:size(z,5)
                    for m = 1:size(z,6)
                        y(1,i,j,k,l,m) = spm_betaln(z(:,i,j,k,l,m));
                    end
                end
            end
        end
    end
end
end 

%--------------------------------------------------------------------------
function [A] = spm_psi(A)
% normalisation of a probability transition rate matrix (columns)
% FORMAT [A] = spm_psi(A)
%
% A  - numeric array
%
% See also: psi.m
% Copyright (C) 2015 Wellcome Trust Centre for Neuroimaging

% Karl Friston
% $Id: spm_psi.m 7300 2018-04-25 21:14:07Z karl $

% normalization of a probability transition rate matrix (columns)
A = bsxfun(@minus, psi(A), psi(sum(A,1)));
end 



