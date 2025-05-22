# CPC Zurich 2025 - Tutorials

The practical tutorials on the 6th day will provide 3-hour, small-group, in-depth and hands-on sessions on a specific modeling approach. All practical sessions will use open-source software packages. 

## Matlab® licenses
Some tutorials require MATLAB which is commercially distributed by MathWorks®. If your tutorial requires MATLAB but you do not have access to a valid license, please contact us at [cpcourse@biomed.ee.ethz.ch](mailto:cpcourse@biomed.ee.ethz.ch)


## Tutorials

### A - Hierarchical Gaussian filter (HGF)
In this tutorial, we will recap the theory behind the Hierarchical Gaussian Filter (HGF) and introduce the model in an accessible way. We will then discuss practical issues when fitting computational models to behavioral data in general and specific to the HGF. We will work through exercises to learn how to analyze data with the HGF using the HGF Toolbox (in Julia and Python).

### B - Active Inference using SPM
In this tutorial, we will review the theory behind active inference and how to implement it within a partially observable Markov decision process (POMDP). We will then do exercises building generative models of common behavioral tasks, learn how to run simulations, and illustrate the useful properties of this modeling framework and when it is and isn't applicable. Finally, we will work through exercises to learn how to fit active inference models to behavioral data and use parameter estimates as individual difference measures in common computational psychiatry contexts. All tutorial exercises will be conducted in MATLAB.

### C - Reinforcement Learning using the hBayesDM Package
In this tutorial, participants will learn how to use a Bayesian package called hBayesDM (supporting R and Python) for modeling various reinforcement learning and decision making (RLDM) tasks. A short overview of (hierarchical) Bayesian modeling will be also provided. Participants will also learn important steps and issues to check when reporting modeling results in publications.

### D - Drift-diffusion model of decision making
In this tutorial, students will learn the theory and practice behind the drift-diffusion model, as it is usually applied to explain behavior (choice, response time, confidence) in simple decision-making tasks. Participants will implement computational simulations to study the properties of the drift-diffusion model, and fit experimental data using MATLAB code provided by the instructor. We will also discuss some of the limitations of the model and common mistakes made when interpreting the model parameters.

### E - Modeling crash-course using the VBA toolbox
In this hands-on tutorial, you will apply computational modeling to a real-life example. Starting from a simple experimental design (delay discounting task), you will learn how to: (a) choose and implement the right model for your task, (b) fit it to empirical data (and get parameter estimates), (c) perform hypothesis testing using model selection, (d) validate your analysis using simulations and diagnostic tools. You will also learn the basics of the VBA-toolbox which contains all the tools to simulate, estimate, and diagnose your models, as well as a collection of ready-to-use models (e.g. Q-learning, DCM). No previous experience with modeling is required, but basic knowledge of MATLAB is recommended.

### F - Machine learning using the PCNtoolkit
Would you like to learn more about modeling individual differences and heterogeneity in psychiatry? In this tutorial, we will abandon the classical patient vs. healthy control framework. You will be guided through how to run an analysis using normative modeling implemented in the PCNtoolkit (using cloud-hosted Python notebooks in Google Colab).

### G - Dynamic causal modeling for EEG
This tutorial will examine specific features of EEG data that can be used to optimize a cell and receptor specific model of brain connectivity. EEG data acquired from an event-related (ERP) visual memory study will be examined. The assumptions and parametrizations of the neural mass models will be explained. Students will learn to use the SPM graphical user interface and to write batch code in MATLAB to perform Dynamic Causal Modeling of EEG.

### H - Dynamic causal modeling for fMRI
In this tutorial you will learn how to use the SPM software to perform a dynamic causal modeling (DCM) analysis in MATLAB. We will first guide you through all steps of a basic DCM analysis of a single subject: data extraction, model setup, model inversion and, finally, inspection of results. We will then proceed to look at a group of subjects. Here, we will focus on model comparison and inspection of model parameters. We will provide a point-by-point recipe on how to perform the analysis. However, it is of advantage if you have worked with neuroimaging (fMRI) data and MATLAB before.

### I - Modeling metacognition using the hMeta-d toolbox
In this tutorial, we will recap the theory underlying the hMeta-d model for quantifying metacognitive efficiency, our ability to monitor and evaluate our own decisions. We will introduce the model in an accessible way, then discuss practical issues when fitting computational models to behavioral data, and go through code examples relevant to computational psychiatry studies using the hMeta-d toolbox (in MATLAB).

### J - Regression DCM
In this tutorial, you will learn how to use the regression dynamic causal modeling (rDCM) toolbox to perform effective (directed) connectivity analyses in whole-brain networks. We will provide you with the necessary theoretical background of the rDCM approach and detail practical aspects that are relevant for whole-brain connectivity analyses. After having laid the foundation, a hands-on part will familiarize you with the code and provide in-depth training on how to apply the model to empirical fMRI data. The goal of this tutorial is to familiarize you with the theoretical and practical aspects of rDCM, which will allow you to seamlessly integrate the approach into your own research. We will provide clear instructions on how to perform the analyses. However, experience with the analysis of fMRI data (already some experience with classical DCM for fMRI would be ideal) as well as experience with Julia or MATLAB are beneficial.

### K - Machine learning for precision psychiatry with NeuroMiner
This interactive, hands-on workshop introduces participants to machine-learning (ML) approaches within the rapidly growing field of precision psychiatry. Combining theoretical concepts with practical exercises, participants will engage directly with [NeuroMiner](https://github.com/neurominer-git/NeuroMiner_1.3), a powerful yet accessible tool developed specifically for clinical neuroscience and neuroimaging research. Attendees will learn about core ML concepts relevant to psychiatric research, such as nested cross-validation, the curse of dimensionality, overfitting prevention, external validation, and explainable AI (XAI). Special attention will be given to specific challenges in neuroimaging and psychiatric data analysis, including site-correction and data fusion techniques. The practical session will guide participants step-by-step through the ML pipeline implementation using NeuroMiner, offering valuable hands-on experience in creating robust and reliable predictive models. Workshop participants will also gain insights into model evaluation strategies through engaging and collaborative exercises. Finally, participants will discuss practical, ethical, and regulatory considerations for integrating ML into clinical psychiatry. Participants will be introduced to the TRIPOD-AI guidelines, ensuring high-quality reporting and communication of ML-driven research results.