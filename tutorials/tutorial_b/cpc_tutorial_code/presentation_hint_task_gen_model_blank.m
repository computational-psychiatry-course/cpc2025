%% Set up POMDP model structure

function MDP = presentation_hint_task_gen_model_blank(Gen_model)

% States and outcomes:

% Hidden State Factor 1: Context
%   - Left machine is more likely to pay out
%   - Right machine is more likely to pay out

% Hidden State Factor 2: Choice states
%   - Start
%   - Ask for the hint
%   - Choose left machine
%   - Choose right machine

% Outcome Modality 1: Hint
%   - No hint
%   - Give hint that the left machine is more likely
%   - Give hint that the right machine is more likely

% Outcome Modality 2: Outcome
%   - Start
%   - Lose
%   - Win

% Outcome Modality 2: Action
%   - Observe start state
%   - Observe get hint state
%   - Observe choose left
%   - Observe choose right



% Number of time points or 'epochs' within a trial: T
% =========================================================================

% Here, we specify 3 time points (T), in which the agent 1) starts in a 'Start'
% state, 2) first moves to either a 'Hint' state or a 'Choose Left' or 'Choose
% Right' slot machine state, and 3) either moves from the Hint state to one
% of the choice states or moves from one of the choice states back to the
% Start state.

T = 3;

% Priors about initial states: D and d
% =========================================================================

%--------------------------------------------------------------------------
% Specify prior probabilities about initial states in the generative 
% process (D) and the agent's generative model (d)
%--------------------------------------------------------------------------

% For the 'context' state factor, we can specify that the 'left better' context 
% (i.e., where the left slot machine is more likely to win) is the true context:

D{1} = [1 0]';  % {'left better','right better'}

% For the 'behavior' state factor, we can specify that the agent always
% begins a trial in the 'start' state (i.e., before choosing to either pick
% a slot machine or first ask for a hint:

D{2} = [1 0 0 0]'; % {'start','hint','choose-left','choose-right'}

%--------------------------------------------------------------------------
% Specify prior beliefs about initial states in the generative model (d)
%--------------------------------------------------------------------------

% Note that these are technically what are called 'Dirichlet concentration
% paramaters', which need not take on values between 0 and 1. These values
% are added to after each trial, based on posterior beliefs about initial
% states. For example, if the agent believed at the end of trial 1 that it 
% was in the 'left better' context, then d{1} on trial 2 would be 
% d{1} = [1.5 0.5]' (although how large the increase in value is after 
% each trial depends on a learning rate). In general, higher values 
% indicate more confidence in one's beliefs about initial states, and 
% entail that beliefs will change more slowly (e.g., the shape of the 
% distribution encoded by d{1} = [25 25]' will change much more slowly 
% than the shape of the distribution encoded by d{1} = [.5 0.5]' with each 
% new observation).

% For context beliefs, we can specify that the agent starts out believing 
% that both contexts are equally likely, but with somewhat low confidence in 
% these beliefs:

d{1} = "TODO";  % {'left better','right better'}

% For behavior beliefs, we can specify that the agent expects with 
% certainty that it will begin a trial in the 'start' state:

d{2} = "TODO"; % {'start','hint','choose-left','choose-right'}

% Assign the floor for learning the d matrix
d_0 = d;


% State-outcome mappings and beliefs: A and a
% =========================================================================

%--------------------------------------------------------------------------
% Specify the probabilities of outcomes given each state in the generative 
% process (A)
%--------------------------------------------------------------------------

% First we specify the mapping from states to observed hints (outcome
% modality 1). Here, the rows correspond to observations, the columns
% correspond to the first state factor (context), and the third dimension
% corresponds to behavior. Each column is a probability distribution
% that must sum to 1.

% We start by specifying that both contexts generate the 'No Hint'
% observation across all behavior states:

Ns = [length(D{1}) length(D{2})]; % number of states in each state factor (2 and 4)

for i = 1:Ns(2) 

    A{1}(:,:,i) = [1 1; % No Hint
                   0 0; % Machine-Left Hint
                   0 0];% Machine-Right Hint
end

% Then we specify that the 'Get Hint' behavior state generates a hint that
% either the left or right slot machine is better, depending on the context
% state. In this case, the hints are accurate with a probability of pHA. 

pHA = 1; % In this example we set this to 1, meaning the advisor will always give an accurate hint

A{1}(:,:,2) = [0     0;      % No Hint
               pHA 1-pHA;    % Machine-Left Hint
               1-pHA pHA];   % Machine-Right Hint


% Now let's specify the agent's learned likelihood matrix for the first
% modality.
% Similar to learning priors over initial states, this simply
% requires specifying a matrix (a) with the same structure as the
% generative process (A), but with Dirichlet concentration parameters that
% can encode beliefs (and confidence in those beliefs) that need not
% match the generative process. Learning then corresponds to
% adding to the values of matrix entries, based on what outcomes were 
% observed when the agent believed it was in a particular state. For
% example, if the agent observed a win while believing it was in the 
% 'left better' context and the 'choose left machine' behavior state,
% the corresponding probability value would increase for that location in
% the state outcome-mapping (i.e., a{2}(3,1,3) might change from .8 to
% 1.8).

% One simple way to set up this matrix is by:
 
% 1. initially identifying it with the generative process 
% 2. multiplying the values by a large number to prevent learning all
%    aspects of the matrix (so the shape of the distribution changes very slowly)
% 3. adjusting the elements you want to differ from the generative process.

% To simulate learning the hint accuracy we
% can specify:

a{1} = A{1}*200;

a{1}(:,:,2) =   ["TODO"    "TODO";     % No Hint
                 "TODO"    "TODO";    % Machine-Left Hint
                 "TODO"    "TODO"];   % Machine-Right Hint


% Next let's specify the likelihood matrix for modality 2: observed wins/losses. The first two
% behavior states ('Start' and 'Get Hint') do not generate either win or
% loss observations in either context:

for i = 1:2

    A{2}(:,:,i) = [1 1;  % Null
                   0 0;  % Loss
                   0 0]; % Win
end
           
% Choosing the left machine (behavior state 3) generates wins with
% probability pWin, which differs depending on the context state (columns):

pWin = .75; % This means that the better side has a 75% chance of producing a win observation
           
A{2}(:,:,3) = [0      0;     % Null        
               1-pWin pWin;  % Loss
               pWin 1-pWin]; % Win

% Choosing the right machine (behavior state 4) generates wins with
% probability pWin, with the reverse mapping to context states from 
% choosing the left machine:
           
A{2}(:,:,4) = [0      0;     % Null
               pWin 1-pWin;  % Loss
               1-pWin pWin]; % Win
           


% Finally, we specify an identity mapping between behavior states and
% observed behaviors, to ensure the agent knows that behaviors were carried
% out as planned. Here, each row corresponds to each behavior state.
           
for i = 1:Ns(2) 

    A{3}(i,:,i) = [1 1];

end

% Specify the likelihood matrix the agent will learn for modality 3:
a{3} = A{3}*200;


    
% Assign the floor for learning the a matrix
a_0 = a;

% Controlled transitions and transition beliefs : B{:,:,u} and b(:,:,u)
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions between hidden states
% under each action (sometimes called 'control states'). 
% Note: By default, these will also be the transitions beliefs 
% for the generative model
%--------------------------------------------------------------------------

% Columns are states at time t. Rows are states at t+1.
% The third dimension of the matrix specifies the action. So B{2}(2,1,2)=1
% means that there is a 100% probability the agent will move to the hint
% state (row 2) if they are in the start state (col 1) and take the hint
% (dim 3)

% The agent cannot control the context state, so there is only 1 'action' 
% (i.e., one entry in third dimension),
% indicating that contexts remain stable within a trial:

B{1}(:,:,1) = [1 0;  % 'Left Better' Context
               0 1]; % 'Right Better' Context
           
% The agent can control the behavior state, and we include 4 possible 
% actions. Note that not all of these transitions will be possible in practice 
% (i.e., the agent would not be able to move to the hint state after
% choosing left machine). Allowable action sequences are specified by the policy
% variable V.

% Move to the Start state from any other state
B{2}(:,:,1) = [1 1 1 1;  % Start State
               0 0 0 0;  % Hint
               0 0 0 0;  % Choose Left Machine
               0 0 0 0]; % Choose Right Machine
           
% Move to the Hint state from any other state
B{2}(:,:,2) = ["TODO" "TODO" "TODO" "TODO";  % Start State
               "TODO" "TODO" "TODO" "TODO";  % Hint
               "TODO" "TODO" "TODO" "TODO";  % Choose Left Machine
               "TODO" "TODO" "TODO" "TODO"]; % Choose Right Machine

% Move to the Choose Left state from any other state
B{2}(:,:,3) = [0 0 0 0;  % Start State
               0 0 0 0;  % Hint
               1 1 1 1;  % Choose Left Machine
               0 0 0 0]; % Choose Right Machine

% Move to the Choose Right state from any other state
B{2}(:,:,4) = [0 0 0 0;  % Start State
               0 0 0 0;  % Hint
               0 0 0 0;  % Choose Left Machine
               1 1 1 1]; % Choose Right Machine        
           
%--------------------------------------------------------------------------
% Specify prior beliefs about state transitions in the generative model
% (b). This is a set of matrices with the same structure as B.
% Note: This is optional, and will simulate learning state transitions if 
% specified.
%--------------------------------------------------------------------------
          
% For this example, we will not simulate learning transition beliefs. 
% But, similar to learning d and a, this just involves accumulating
% Dirichlet concentration parameters. Here, transition beliefs are updated
% after each trial when the agent believes it was in a given state at time
% t and and another state at t+1.

% Preferred outcomes: C and c
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the 'prior preferences', encoded here as log
% probabilities. 
%--------------------------------------------------------------------------

% One matrix per outcome modality. Each row is an observation, and each
% columns is a time point. Negative values indicate lower preference,
% positive values indicate a high preference. Stronger preferences promote
% risky choices and reduced information-seeking.

% We can start by setting a 0 preference for all outcomes:

No = [size(A{1},1) size(A{2},1) size(A{3},1)]; % number of outcomes in 
                                               % each outcome modality

C{1}      = zeros(No(1),T); % Hints
C{2}      = zeros(No(2),T); % Wins/Losses
C{3}      = zeros(No(3),T); % Observed Behaviors

% Then we can specify a 'loss aversion' magnitude (la) at time points 2 
% and 3, and a 'reward seeking' (or 'risk-seeking') magnitude (rs). Here,
% rs is divided by 2 at the third time point to encode a smaller win ($2
% instead of $4) if taking the hint before choosing a slot machine.
rs = 4;
la = -1;

C{2}(:,:) =    [0  0   0 ;  % Null
                0  "TODO" "TODO" ;  % Loss
                0  "TODO" "TODO"];  % win
            
%--------------------------------------------------------------------------
% One can also optionally choose to simulate preference learning by
% specifying a Dirichlet distribution over preferences (c). 
%--------------------------------------------------------------------------

% This will not be simulated here. However, this works by increasing the
% preference magnitude for an outcome each time that outcome is observed.
% The assumption here is that preferences naturally increase for entering
% situations that are more familiar.

% Allowable policies: U or V. 
%==========================================================================

%--------------------------------------------------------------------------
% Each policy is a sequence of actions over time that the agent can 
% consider. 
%--------------------------------------------------------------------------

% For our simulations, we will specify V, where rows correspond to time 
% points and should be length T-1 (here, 2 transitions, from time point 1
% to time point 2, and time point 2 to time point 3):

NumPolicies = 5; % Number of policies
NumFactors = 2; % Number of state factors

V         = ones(T-1,NumPolicies,NumFactors);

V(:,:,1) = [1 1 1 1 1;
            1 1 1 1 1]; % Context state (i.e., left better or right better) is not controllable

V(:,:,2) = [1 2 2 3 4; % action state chosen at T=1
            1 3 4 1 1]; % action state chosen at T=2
        
% For V(:,:,2), columns left to right indicate policies allowing: 
% 1. staying in the start state 
% 2. taking the hint then choosing the left machine
% 3. taking the hint then choosing the right machine
% 4. choosing the left machine right away (then returning to start state)
% 5. choosing the right machine right away (then returning to start state)


% Habits: E and e. 
%==========================================================================

%--------------------------------------------------------------------------
% Optional: a columns vector with one entry per policy, indicating the 
% prior probability of choosing that policy (i.e., independent of other 
% beliefs). 
%--------------------------------------------------------------------------

% We will not equip our agent with habits with any starting habits 
% (flat distribution over policies):

E = [1 1 1 1 1]';

% To incorporate habit learning, where policies become more likely after 
% each time they are chosen, we can also specify concentration parameters
% by specifying e:

 e = [1 1 1 1 1]';

 % Assign the floor for learning the e matrix
 e_0 = e;

% Additional optional parameters. 
%==========================================================================

% Eta: learning rate (0-1) controlling the magnitude of concentration parameter
% updates after each trial (if learning is enabled).

     eta = 1; % Default (maximum) learning rate
     
% Omega: forgetting rate (0-1) controlling the magnitude of reduction in concentration
% parameter values after each trial (if learning is enabled).

     omega = 0; % Default value indicating there is no forgetting 

% Beta: Inverse Expected precision of expected free energy (G) over policies (a 
% positive value, with lower values indicating higher expected precision).
% Higher values increase the influence of habits (E) and otherwise make
% policy selection less deteriministic.

     beta = 1; % By default this is set to 1, but try increasing its value 
               %  see how it affects model behavior

% Alpha: An 'inverse temperature' or 'action precision' parameter that 
% controls how much randomness there is when selecting actions (e.g., how 
% often the agent might choose not to take the hint, even if the model 
% assigned the highest probability to that action. This is a positive 
% number, where higher values indicate less randomness. Here we set this to 
% a fairly high value:

    alpha = 1; % fairly low randomness in action selection

%% Define POMDP Structure
%==========================================================================

mdp.T = T;                    % Number of time steps
mdp.V = V;                    % allowable (deep) policies

mdp.A = A;                    % state-outcome mapping
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = D;                    % priors over initial states
mdp.d = d;                    % enable learning priors over initial states
mdp.d_0 = d_0;                % Assign floor for learning context

if Gen_model == 1
    mdp.E = E;                % prior over policies
elseif Gen_model == 2
    mdp.a = a;                % enable learning state-outcome mappings
    mdp.a_0 = a_0;            % Assign floor for learning likelihood
    mdp.e = e;                % enable learning of prior over policies
    mdp.e_0 = e_0;            % Assign floor for learning habits

end 

mdp.eta = eta;                % learning rate
mdp.omega = omega;            % forgetting rate
mdp.alpha = alpha;            % action precision
mdp.beta = beta;              % expected free energy precision

%respecify for use in inversion script (specific to this tutorial example)
mdp.NumPolicies = NumPolicies; % Number of policies
mdp.NumFactors = NumFactors; % Number of state factors
    
   
% We can add labels to states, outcomes, and actions for subsequent plotting:

label.factor{1}   = 'contexts';   label.name{1}    = {'left-better','right-better'};
label.factor{2}   = 'choice states';     label.name{2}    = {'start','hint','choose left','choose right'};
label.modality{1} = 'hint';    label.outcome{1} = {'null','left hint','right hint'};
label.modality{2} = 'win/lose';  label.outcome{2} = {'null','lose','win'};
label.modality{3} = 'observed action';  label.outcome{3} = {'start','hint','choose left','choose right'};
label.action{2} = {'start','hint','left','right'};
mdp.label = label;

MDP = mdp;

end